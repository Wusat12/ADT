import argparse
import torch
from torch.utils.data import DataLoader
from time import time
from preprocess import create_transform, load_dataset
from evaluate import evaluate_model
from pathlib import Path

# Only run this file directly
assert __name__ == "__main__"

DATASETS = ["enst+mdb", "egmd", "slakh", "adtof_yt", "sadtp"]

parser = argparse.ArgumentParser("test.py")
parser.add_argument("device", help="The device to run experiments on", type=str, default="cuda:0", nargs="?")
parser.add_argument("--path", help="The path to the model to test, relative to the root directory", required=True)
parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=DATASETS, help="Datasets to test on (missing converted files are skipped)")
args = parser.parse_args()

root_dir = Path(__file__).resolve().parent
model_dir = root_dir / args.path
data_dir = root_dir / "data"
assert model_dir.exists(), "Model path is not valid"

print(f"Main: Can use CUDA: {torch.cuda.is_available()}")
device = args.device
seed = int(time())
batch_size = 128

# Load the model we want to test on
config = torch.load(model_dir / "config.pt", weights_only=False)
state_dict = torch.load(model_dir / "model.pt", map_location="cpu")
representation = config.get("representation", "logmel")
print(f"Frontend representation: {representation}")

model = config["Model"](**config["parameters"])
model.load_state_dict(state_dict)
transforms = create_transform(mean=config["transforms"]["mean"], std=config["transforms"]["std"], channels_last=True)

dataset_path = {
    "enst+mdb": data_dir / "ENST+MDB",
    "egmd": data_dir / "e-gmd-v1.0.0",
    "slakh": data_dir / "slakh2100_flac_redux",
    "adtof_yt": data_dir / "adtof",
    "sadtp": data_dir / "SADTP",
}

with open(model_dir / "tests.txt", "w") as output:
    for dataset in args.datasets:
        test_path = dataset_path[dataset] / f"{dataset}_{representation}_test.pt"
        if not test_path.exists():
            print(f"Skipping {dataset}: {test_path.name} not found")
            continue

        test_loader = DataLoader(load_dataset(test_path), batch_size=batch_size, num_workers=4, pin_memory=True)
        test_f1_micro, test_f1_macro, test_f1_class = evaluate_model(model, test_loader=test_loader, transforms=transforms, seed=seed, device=device)

        for out in (None, output):
            print(f" ---------- Evaluation on {dataset.upper()} ---------- ", file=out)
            print(f"Micro F1: {test_f1_micro.item():.4f}", file=out)
            print(f"Macro F1: {test_f1_macro.item():.4f}", file=out)
            print(f"Class F1: {[f'{f1.item():.4f}' for f in test_f1_class]}", file=out)
