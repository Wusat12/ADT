import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from ray import init, tune, train
from ray.tune.search.optuna import OptunaSearch
from time import time
from models import RNN, CNN, ConvolutionalRNN, ConvolutionalTransformer, VisionTransformer
from preprocess import compute_normalization, create_transform
from evaluate import evaluate_model
from train import train_model
from pathlib import Path

# Only run this file directly
assert __name__ == "__main__"

# Declare an argument parser for this file
parser = argparse.ArgumentParser("run.py")
parser.add_argument("device", help="The device to run experiments on", type=str, default="cuda:0", nargs="?")
parser.add_argument("--model", choices=["rnn", "cnn", "crnn", "ct", "vit"], help="The model to train", required=True)
parser.add_argument("--dataset", choices=["enst+mdb", "egmd", "slakh", "adtof_yt"], help="The dataset to train on", nargs="+", required=True)
parser.add_argument("--num_samples", type=int, help="Number of samples for Optuna RayTune", required=False, default=15)
parser.add_argument("--early_stop", type=int, help="Number of epochs with stagnating validation loss before early stopping", required=False, default=15)
args = parser.parse_args()

# Extract the absolute path of the data directory
root_dir = Path(__file__).resolve().parent
data_dir = root_dir / "data"

# Initialize a Ray instance
temp_dir = Path.home().resolve() / ".ray_temp"
init(num_gpus=1, num_cpus=5, _temp_dir=temp_dir.as_posix())

# ----------------------------------------------------------------------------------------------------------------

print(f"Main: Can use CUDA: {torch.cuda.is_available()}")
device = args.device
seed = int(time())

Model = {
    "rnn": RNN, 
    "cnn": CNN, 
    "crnn": ConvolutionalRNN, 
    "ct": ConvolutionalTransformer, 
    "vit": VisionTransformer,
    }[args.model]

dataset_paths = [{
    "enst+mdb": data_dir / "ENST+MDB",
    "egmd": data_dir / "e-gmd-v1.0.0",
    "slakh": data_dir / "slakh2100_flac_redux",
    "adtof_yt": data_dir / "adtof",
}[dataset] for dataset in args.dataset]

study = "Architecture" if len(args.dataset) == 1 else "Dataset"
experiment = Model.name

num_samples = args.num_samples
num_epochs = 100

batch_size = 128

train_paths = [dataset_path / (dataset + "_train.pt") for dataset_path, dataset in zip(dataset_paths, args.dataset)]
val_paths = [dataset_path / (dataset + "_validation.pt") for dataset_path, dataset in zip(dataset_paths, args.dataset)]
test_paths = [dataset_path / (dataset + "_test.pt") for dataset_path, dataset in zip(dataset_paths, args.dataset)]

feature_mean, feature_std = compute_normalization(train_paths, device=device)

print(f"Traning data has a mean of: {feature_mean}, and a std of: {feature_std}")

config = {
    "num_epochs": num_epochs,
    "batch_size": batch_size,

    "train_paths": train_paths,
    "val_paths": val_paths,

    "transforms": {
        "mean": feature_mean,
        "std": feature_std
    },

    "lr": tune.loguniform(1e-4, 5e-3),
    "weight_decay": tune.loguniform(1e-6, 1e-2),
    "optimizer": optim.AdamW,

    "Model": Model,
    "parameters": Model.hyperparameters,

    "device": device,
    "seed": seed,
}

# Run the experiments
tuner = tune.Tuner(
    tune.with_resources(
        trainable=train_model,
        resources={"gpu": 1, "accelerator_type:A100": 1}
    ),
    param_space=config,
    tune_config=tune.TuneConfig(
        num_samples=num_samples,
        metric="best_epoch/Micro F1",
        mode="max",
        search_alg=OptunaSearch(
            metric="best_epoch/Micro F1",
            mode="max"
        )
    ),
    run_config=train.RunConfig(
        stop={"epochs_since_improvement": args.early_stop},
        checkpoint_config=train.CheckpointConfig(num_to_keep=1),
        verbose=2
    )
)
results = tuner.fit()

# Print the results
best_result = results.get_best_result("Micro F1", mode="max", scope="all")
print(f"Best result config: {best_result.config}")
print(f"Best result validation loss: {best_result.metrics['best_epoch']['Validation Loss']}")
print(f"Best result validation micro F1: {best_result.metrics['best_epoch']['Micro F1']}")
print(f"Best result validation macro F1: {best_result.metrics['best_epoch']['Macro F1']}")
print(f"Best result validation class F1: {best_result.metrics['best_epoch']['Class F1']}")

# Load the state_dict of the best performing model
best_checkpoint = best_result.get_best_checkpoint("Micro F1", mode="max")
state_dict = torch.load(Path(best_checkpoint.path) / "model.pt")

# Store the best performing model, config and its metrics to study/experiment path
study_path = (root_dir / "study" / study / experiment / "+".join(args.dataset).upper().replace("_", "-"))
study_path.mkdir(parents=True, exist_ok=True)
torch.save(state_dict, study_path / "model.pt")
torch.save(best_result.config, study_path / "config.pt")
best_result.metrics_dataframe.to_csv(study_path / "metrics.csv")

# Load the best performing model
model = Model(**best_result.config["parameters"])
model.load_state_dict(state_dict)

# Create a test dataloader and preprocessing transforms
test_loader = DataLoader(ConcatDataset(map(torch.load, test_paths)), batch_size=batch_size, num_workers=4, pin_memory=True)
transforms = create_transform(mean=feature_mean, std=feature_std, channels_last=True)

# And evaluate it
test_f1_micro, test_f1_macro, test_f1_class = evaluate_model(model, test_loader=test_loader, transforms=transforms, seed=seed, device=device)

print(" ---------- Evaluation of best perfoming model ---------- ")
print(f"Micro F1: {test_f1_micro.item():.4f}")
print(f"Macro F1: {test_f1_macro.item():.4f}")
print(f"Class F1: {[f'{test_f1.item():.4f}' for test_f1 in test_f1_class]}")

# Finally, write the results to an output file
with open(study_path / "results.txt", "w") as output:
    print(f"Best result config: {best_result.config}", file=output)
    print(f"Best result final validation loss: {best_result.metrics['best_epoch']['Validation Loss']}", file=output)
    print(f"Best result final validation micro F1: {best_result.metrics['best_epoch']['Micro F1']}", file=output)
    print(f"Best result final validation macro F1: {best_result.metrics['best_epoch']['Macro F1']}", file=output)
    print(f"Best result final validation class F1: {best_result.metrics['best_epoch']['Class F1']}", file=output)
    print(" ---------- Evaluation of best perfoming model ---------- ", file=output)
    print(f"Micro F1: {test_f1_micro.item():.4f}", file=output)
    print(f"Macro F1: {test_f1_macro.item():.4f}", file=output)
    print(f"Class F1: {[f'{test_f1.item():.4f}' for test_f1 in test_f1_class]}", file=output)