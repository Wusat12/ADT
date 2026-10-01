import torch
import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse

""" Run this file to turn a stored Tensorflow dataset into a stored PyTorch dataset """

# Declare an argument parser for this file
parser = argparse.ArgumentParser("convert_adtof.py")
parser.add_argument("--directory", help="The outer directory for the ADTOF-YT dataset", required=False, default="adtof")
args = parser.parse_args()

if __name__ == "__main__":
    # Declare the path to the dataset directory
    path = Path(__file__).resolve().parent.parent / "data" / args.directory

    # Split into Train, Validation, Test
    for split in ["train", "validation", "test"]:

        print("\033[96m", f"    Loading TensorFlow {split} dataset from disk", "\033[0m", sep="")

        # Load the dataset
        tf_dataset = tf.data.Dataset.load(str(path / f"adtof_yt_{split}"))

        print("\033[96m", "     Loading data into lists", "\033[0m", sep="")

        # Store all features and labels
        features, labels = [], []
        for data, label in tf_dataset.as_numpy_iterator():
            features.append(data["x"])
            labels.append(label)
        features, labels = np.array(features), np.array(labels)

        print("\033[96m", "     Creating tensor datasets", "\033[0m", sep="")

        # Turn them into a Pytorch tensor dataset
        dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))

        # And store the dataset to the disk under the first path
        new_path = (path / f"adtof_yt_{split}").with_suffix(".pt")
        print("\033[96m", "     Storing ", "\033[0m", f"{new_path.name}", "\033[96m", " to disk", "\033[0m", sep="")
        torch.save(dataset, new_path)

        print("\033[95m", "     Finished!", "\033[0m", sep="")
    
        # Load dataset and verify that everything is correct
        dataset = torch.load(new_path)
        print("\033[92m", "     Final dataset contains ", "\033[0m", len(dataset), "\033[92m", " entries", "\033[0m", sep="")
        print("\033[92m", "     Each entry has features of shape: ", "\033[0m", dataset[0][0].shape, "\033[92m", ", and labels of shape: ", "\033[0m", dataset[0][1].shape, sep="")
        print("\033[92m", "     Each class has a frequency of: ", "\033[0m", dataset[:][1].round().sum(dim=(0, 1)), sep="")

    # Verify that dataloaders work
    dataloader = DataLoader(dataset, batch_size=16)
    for features, labels in dataloader:
        print("\033[92m", "Batched entry in dataloader has features of shape: ", "\033[0m", features.shape, "\033[92m", ", and labels of shape: ", "\033[0m", labels.shape, sep="")
        break