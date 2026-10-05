import os

# Keep Python/Ray temporary files on E: instead of C:.
os.environ["TEMP"] = "E:\\temp"
os.environ["TMP"] = "E:\\temp"


import argparse
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset

from ray import init, tune, train
from ray.tune.search.optuna import OptunaSearch

from models import CNN

from preprocess import (
    compute_normalization,
    create_transform,
    load_dataset,
)

from evaluate import evaluate_model

from train import train_model


# Only run this file directly.
assert __name__ == "__main__"


# ----------------------------------------------------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    "run.py"
)

parser.add_argument(
    "device",
    help="The device to run experiments on",
    type=str,
    default="cuda:0",
    nargs="?",
)

parser.add_argument(
    "--dataset",
    choices=[
        "enst+mdb",
        "egmd",
        "slakh",
        "adtof_yt",
    ],
    help="The dataset to train on",
    nargs="+",
    required=True,
)

parser.add_argument(
    "--num_samples",
    type=int,
    help="Number of samples for Optuna RayTune",
    required=False,
    default=15,
)

parser.add_argument(
    "--early_stop",
    type=int,
    help=(
        "Number of epochs with stagnating "
        "validation performance before early stopping"
    ),
    required=False,
    default=15,
)

parser.add_argument(
    "--representation",
    choices=[
        "logmel",
        "pcen",
    ],
    default="logmel",
    help=(
        "Frontend representation used "
        "to create the dataset"
    ),
)

args = parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------
# Repository paths
# ----------------------------------------------------------------------------------------------------------------

root_dir = Path(
    __file__
).resolve().parent

data_dir = (
    root_dir / "data"
)


# ----------------------------------------------------------------------------------------------------------------
# Ray initialization
# ----------------------------------------------------------------------------------------------------------------

temp_dir = Path(
    "E:/ray_temp"
)

temp_dir.mkdir(
    parents=True,
    exist_ok=True,
)


ray_results_dir = Path(
    "E:/ray_results"
)

ray_results_dir.mkdir(
    parents=True,
    exist_ok=True,
)


python_temp_dir = Path(
    "E:/temp"
)

python_temp_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# Use one CPU for the current Windows CPU test.
num_ray_cpus = 1


init(
    num_gpus=int(
        torch.cuda.is_available()
    ),
    num_cpus=num_ray_cpus,
    _temp_dir=temp_dir.as_posix(),
)


print(
    f"Main: Can use CUDA: "
    f"{torch.cuda.is_available()}"
)

print(
    f"Python TEMP: "
    f"{os.environ['TEMP']}"
)

print(
    f"Python TMP: "
    f"{os.environ['TMP']}"
)

print(
    f"Ray temp directory: "
    f"{temp_dir}"
)

print(
    f"Ray results directory: "
    f"{ray_results_dir}"
)


# ----------------------------------------------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = args.device
else:
    device = "cpu"


# ----------------------------------------------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------------------------------------------

seed = 42


# ----------------------------------------------------------------------------------------------------------------
# Downstream model
# ----------------------------------------------------------------------------------------------------------------

# The same CNN is used for Log-Mel and PCEN.
Model = CNN


# ----------------------------------------------------------------------------------------------------------------
# Dataset paths
# ----------------------------------------------------------------------------------------------------------------

dataset_directories = {
    "enst+mdb":
        data_dir / "ENST+MDB",

    "egmd":
        data_dir / "e-gmd-v1.0.0",

    "slakh":
        data_dir / "slakh2100_flac_redux",

    "adtof_yt":
        data_dir / "adtof",
}


dataset_paths = [
    dataset_directories[dataset]
    for dataset in args.dataset
]


study = (
    "Architecture"
    if len(args.dataset) == 1
    else "Dataset"
)


suffix = (
    f"_{args.representation}"
)


experiment = Model.name


# ----------------------------------------------------------------------------------------------------------------
# Training parameters
# ----------------------------------------------------------------------------------------------------------------

num_samples = args.num_samples

# Current test configuration.
num_epochs = 10

# Current test configuration.
batch_size = 16


# ----------------------------------------------------------------------------------------------------------------
# Representation-specific dataset files
# ----------------------------------------------------------------------------------------------------------------

train_paths = [
    dataset_path
    / f"{dataset}{suffix}_train.pt"

    for dataset_path, dataset
    in zip(
        dataset_paths,
        args.dataset,
    )
]


validation_paths = [
    dataset_path
    / f"{dataset}{suffix}_validation.pt"

    for dataset_path, dataset
    in zip(
        dataset_paths,
        args.dataset,
    )
]


test_paths = [
    dataset_path
    / f"{dataset}{suffix}_test.pt"

    for dataset_path, dataset
    in zip(
        dataset_paths,
        args.dataset,
    )
]


# ----------------------------------------------------------------------------------------------------------------
# Check dataset files
# ----------------------------------------------------------------------------------------------------------------

for path in (
    train_paths
    + validation_paths
    + test_paths
):

    if not path.is_file():

        raise FileNotFoundError(
            "Required dataset file was not found:\n"
            f"{path}"
        )


# ----------------------------------------------------------------------------------------------------------------
# Compute normalization statistics
# ----------------------------------------------------------------------------------------------------------------

feature_mean, feature_std = (
    compute_normalization(
        train_paths,
        device=device,
    )
)


print(
    f"Training data has a mean of: "
    f"{feature_mean}, "
    f"and a std of: "
    f"{feature_std}"
)


# ----------------------------------------------------------------------------------------------------------------
# Ray Tune configuration
# ----------------------------------------------------------------------------------------------------------------

# These keys are shared directly with train.py.

config = {

    "num_epochs":
        num_epochs,

    "batch_size":
        batch_size,

    "train_paths":
        train_paths,

    "val_paths":
        validation_paths,

    "transforms": {
        "mean":
            feature_mean,

        "std":
            feature_std,
    },

    "lr":
        tune.loguniform(
            1e-4,
            5e-3,
        ),

    "weight_decay":
        tune.loguniform(
            1e-6,
            1e-2,
        ),

    "optimizer":
        optim.AdamW,

    "Model":
        Model,

    "parameters":
        Model.hyperparameters,

    "device":
        device,

    "seed":
        seed,

    "representation":
        args.representation,

    "num_workers":
        0,

    "early_stop":
        args.early_stop,
}


# ----------------------------------------------------------------------------------------------------------------
# Ray resources
# ----------------------------------------------------------------------------------------------------------------

if torch.cuda.is_available():

    resources = {
        "gpu": 1,
    }

else:

    resources = {
        "cpu": 1,
    }


# ----------------------------------------------------------------------------------------------------------------
# Short Ray trial directory name
# ----------------------------------------------------------------------------------------------------------------

def trial_dirname_creator(trial):

    return (
        f"t{trial.trial_id}"
    )


# ----------------------------------------------------------------------------------------------------------------
# Ray Tune
# ----------------------------------------------------------------------------------------------------------------

tuner = tune.Tuner(

    tune.with_resources(
        trainable=train_model,
        resources=resources,
    ),

    param_space=config,

    tune_config=tune.TuneConfig(

        num_samples=num_samples,

        metric="best_epoch/Micro F1",

        mode="max",

        search_alg=OptunaSearch(

            metric="best_epoch/Micro F1",

            mode="max",
        ),

        trial_dirname_creator=(
            trial_dirname_creator
        ),
    ),

    run_config=train.RunConfig(

        name="adt",

        storage_path=(
            ray_results_dir.as_posix()
        ),

        stop={
            "epochs_since_improvement":
                args.early_stop,
        },

        checkpoint_config=(
            train.CheckpointConfig(
                num_to_keep=1,
            )
        ),

        verbose=2,
    ),
)


# ----------------------------------------------------------------------------------------------------------------
# Start experiment
# ----------------------------------------------------------------------------------------------------------------

results = tuner.fit()


# ----------------------------------------------------------------------------------------------------------------
# Inspect completed trials
# ----------------------------------------------------------------------------------------------------------------

print(
    f"\nRay completed "
    f"{len(results)} trial(s)."
)


if len(results) == 0:

    raise RuntimeError(
        "Ray Tune completed without "
        "producing any trials."
    )


# ----------------------------------------------------------------------------------------------------------------
# Find best result
# ----------------------------------------------------------------------------------------------------------------

best_result = (
    results.get_best_result(
        metric="best_epoch/Micro F1",
        mode="max",
    )
)


# ----------------------------------------------------------------------------------------------------------------
# Diagnostic output
# ----------------------------------------------------------------------------------------------------------------

print(
    "\nBest trial ID:",
    best_result.metrics.get(
        "trial_id"
    ),
)

print(
    "Best trial path:",
    best_result.path,
)

print(
    "\nMetrics reported by "
    "the best trial:"
)

for key, value in sorted(
    best_result.metrics.items()
):

    print(
        f"  {key}: {value}"
    )


# ----------------------------------------------------------------------------------------------------------------
# Extract best metrics
# ----------------------------------------------------------------------------------------------------------------

best_validation_loss = (
    best_result.metrics.get(
        "best_epoch/Validation Loss"
    )
)

best_validation_micro_f1 = (
    best_result.metrics.get(
        "best_epoch/Micro F1"
    )
)

best_validation_macro_f1 = (
    best_result.metrics.get(
        "best_epoch/Macro F1"
    )
)

best_validation_class_f1 = (
    best_result.metrics.get(
        "best_epoch/Class F1"
    )
)


# ----------------------------------------------------------------------------------------------------------------
# Print best result
# ----------------------------------------------------------------------------------------------------------------

print(
    f"\nBest result config: "
    f"{best_result.config}"
)

print(
    f"Best result validation loss: "
    f"{best_validation_loss}"
)

print(
    f"Best result validation micro F1: "
    f"{best_validation_micro_f1}"
)

print(
    f"Best result validation macro F1: "
    f"{best_validation_macro_f1}"
)

print(
    f"Best result validation class F1: "
    f"{best_validation_class_f1}"
)


# ----------------------------------------------------------------------------------------------------------------
# Check objective
# ----------------------------------------------------------------------------------------------------------------

if best_validation_micro_f1 is None:

    raise RuntimeError(
        "The selected Ray trial does not contain "
        "'best_epoch/Micro F1'.\n\n"
        "The complete metric dictionary was "
        "printed above."
    )


# ----------------------------------------------------------------------------------------------------------------
# Load best checkpoint
# ----------------------------------------------------------------------------------------------------------------

best_checkpoint = (
    best_result.get_best_checkpoint(
        metric="best_epoch/Micro F1",
        mode="max",
    )
)


if best_checkpoint is None:

    raise RuntimeError(
        "Ray selected a best trial but did not "
        "return a checkpoint for that trial.\n\n"
        f"Trial path:\n"
        f"{best_result.path}"
    )


print(
    "\nBest checkpoint:",
    best_checkpoint,
)


# ----------------------------------------------------------------------------------------------------------------
# Load model state dictionary
# ----------------------------------------------------------------------------------------------------------------

with best_checkpoint.as_directory() as checkpoint_dir:

    checkpoint_path = (
        Path(checkpoint_dir)
        / "model.pt"
    )

    if not checkpoint_path.is_file():

        raise FileNotFoundError(
            "Ray checkpoint was found, but "
            "model.pt does not exist:\n"
            f"{checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location="cpu",
    )


# ----------------------------------------------------------------------------------------------------------------
# Experiment output directory
# ----------------------------------------------------------------------------------------------------------------

study_path = (
    root_dir
    / "experiments"
    / args.representation
    / study
    / experiment
    / "+".join(
        args.dataset
    ).upper().replace(
        "_",
        "-",
    )
)


study_path.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------------------------------------------------------------------------
# Save best model
# ----------------------------------------------------------------------------------------------------------------

torch.save(
    state_dict,
    study_path / "model.pt",
)


# ----------------------------------------------------------------------------------------------------------------
# Save configuration
# ----------------------------------------------------------------------------------------------------------------

torch.save(
    best_result.config,
    study_path / "config.pt",
)


# ----------------------------------------------------------------------------------------------------------------
# Save Ray metrics
# ----------------------------------------------------------------------------------------------------------------

best_result.metrics_dataframe.to_csv(
    study_path / "metrics.csv"
)


# ----------------------------------------------------------------------------------------------------------------
# Create best model
# ----------------------------------------------------------------------------------------------------------------

model = Model(
    **best_result.config[
        "parameters"
    ]
)


model.load_state_dict(
    state_dict
)


model = model.to(
    device
)


# ----------------------------------------------------------------------------------------------------------------
# Test DataLoader
# ----------------------------------------------------------------------------------------------------------------

test_loader = DataLoader(

    ConcatDataset(
        map(
            load_dataset,
            test_paths,
        )
    ),

    batch_size=batch_size,

    num_workers=0,

    pin_memory=(
        torch.cuda.is_available()
    ),
)


# ----------------------------------------------------------------------------------------------------------------
# Test preprocessing
# ----------------------------------------------------------------------------------------------------------------

transforms = create_transform(
    mean=feature_mean,
    std=feature_std,
    channels_last=True,
)


# ----------------------------------------------------------------------------------------------------------------
# Evaluate best model
# ----------------------------------------------------------------------------------------------------------------

test_f1_micro, test_f1_macro, test_f1_class = (
    evaluate_model(
        model,

        test_loader=test_loader,

        transforms=transforms,

        seed=seed,

        device=device,
    )
)


# ----------------------------------------------------------------------------------------------------------------
# Print test results
# ----------------------------------------------------------------------------------------------------------------

print(
    " ---------- Evaluation of best "
    "performing model ---------- "
)

print(
    f"Micro F1: "
    f"{test_f1_micro.item():.4f}"
)

print(
    f"Macro F1: "
    f"{test_f1_macro.item():.4f}"
)

print(
    "Class F1: "
    f"{[f'{test_f1.item():.4f}' for test_f1 in test_f1_class]}"
)


# ----------------------------------------------------------------------------------------------------------------
# Save final results
# ----------------------------------------------------------------------------------------------------------------

with open(
    study_path / "results.txt",
    "w",
) as output:

    print(
        f"Representation: "
        f"{args.representation}",
        file=output,
    )

    print(
        f"Dataset: "
        f"{args.dataset}",
        file=output,
    )

    print(
        f"Best result config: "
        f"{best_result.config}",
        file=output,
    )

    print(
        f"Best result validation loss: "
        f"{best_validation_loss}",
        file=output,
    )

    print(
        f"Best result validation micro F1: "
        f"{best_validation_micro_f1}",
        file=output,
    )

    print(
        f"Best result validation macro F1: "
        f"{best_validation_macro_f1}",
        file=output,
    )

    print(
        f"Best result validation class F1: "
        f"{best_validation_class_f1}",
        file=output,
    )

    print(
        " ---------- Evaluation of best "
        "performing model ---------- ",
        file=output,
    )

    print(
        f"Micro F1: "
        f"{test_f1_micro.item():.4f}",
        file=output,
    )

    print(
        f"Macro F1: "
        f"{test_f1_macro.item():.4f}",
        file=output,
    )

    print(
        "Class F1: "
        f"{[f'{test_f1.item():.4f}' for test_f1 in test_f1_class]}",
        file=output,
    )


# ----------------------------------------------------------------------------------------------------------------
# Finished
# ----------------------------------------------------------------------------------------------------------------

print(
    "\nExperiment completed."
)

print(
    f"Results saved to:\n"
    f"{study_path}"
)
