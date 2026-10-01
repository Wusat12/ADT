import argparse
import torch
from torch.utils.data import DataLoader
from time import time
from preprocess import create_transform
from evaluate import evaluate_model
from pathlib import Path

# Only run this file directly
assert __name__ == "__main__"

# Declare an argument parser for this file
parser = argparse.ArgumentParser("run.py")
parser.add_argument("device", help="The device to run experiments on", type=str, default="cuda:0", nargs="?")
parser.add_argument("--path", help="The path to the model to test, relative to the root directory", required=True)
args = parser.parse_args()

# Extract the absolute path of the directories
root_dir = Path(__file__).resolve().parent
model_dir = root_dir / args.path
data_dir = root_dir / "data"

assert model_dir.exists(), "Model path is not valid"

# ----------------------------------------------------------------------------------------------------------------

print(f"Main: Can use CUDA: {torch.cuda.is_available()}")
device = args.device
seed = int(time())

batch_size = 128

# Load the model we want to test on
config = torch.load(model_dir / "config.pt")
state_dict = torch.load(model_dir / "model.pt")
model = config["Model"](**config["parameters"])
model.load_state_dict(state_dict)

# Create the transforms
transforms = create_transform(mean=config["transforms"]["mean"], std=config["transforms"]["std"], channels_last=True)

# Then, for each of the datasets
dataset_path = {
    "enst+mdb": data_dir / "ENST+MDB",
    "egmd": data_dir / "e-gmd-v1.0.0",
    "slakh": data_dir / "slakh2100_flac_redux",
    "adtof_yt": data_dir / "adtof",
    "sadtp": data_dir/ "SADTP"
}
with open(model_dir / "tests.txt", "w") as output:
    for dataset in ["enst+mdb", "egmd", "slakh", "adtof_yt", "sadtp"]:
        # Test and evaluate predictions
        test_loader = DataLoader(torch.load(dataset_path[dataset] / (dataset + "_test.pt")), batch_size=batch_size, num_workers=4, pin_memory=True)
        test_f1_micro, test_f1_macro, test_f1_class = evaluate_model(model, test_loader=test_loader, transforms=transforms, seed=seed, device=device)

        print(f" ---------- Evaluation on {dataset.upper()} ---------- ")
        print(f"Micro F1: {test_f1_micro.item():.4f}")
        print(f"Macro F1: {test_f1_macro.item():.4f}")
        print(f"Class F1: {[f'{test_f1.item():.4f}' for test_f1 in test_f1_class]}")

        # Also, write the results to an output file
        print(f" ---------- Evaluation on {dataset.upper()} ---------- ", file=output)
        print(f"Micro F1: {test_f1_micro.item():.4f}", file=output)
        print(f"Macro F1: {test_f1_macro.item():.4f}", file=output)
        print(f"Class F1: {[f'{test_f1.item():.4f}' for test_f1 in test_f1_class]}", file=output)