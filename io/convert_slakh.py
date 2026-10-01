import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import argparse

from load import readAudio, readMidi
from mapping import MIDI_MAPPING

""" Run this file to turn Slakh2100-redux into a stored PyTorch dataset """

# Declare an argument parser for this file
parser = argparse.ArgumentParser("convert_slakh.py")
parser.add_argument("--directory", help="The outer directory name for the Slakh dataset", required=False, default="slakh2100_flac_redux")
args = parser.parse_args()

vocabulary = set()

if __name__ == "__main__":
    # Declare the path to the dataset directory
    path = Path(__file__).resolve().parent.parent / "data" / args.directory

    # Split into Train, Validation, Test
    for split in ["train", "validation", "test"]:

        print("\033[96m", f"Loading {split} data into lists", "\033[0m", sep="")

        data, labels = [], []
        for track in (path / split).iterdir():
            with open((track / "metadata").with_suffix(".yaml")) as f:
                metadata = yaml.safe_load(f)
            
            drum_stem = next(stem for stem, info in metadata["stems"].items() if info["is_drum"])

            audio_path = (track / "mix").with_suffix(".flac")
            midi_path = (track / "MIDI" / drum_stem).with_suffix(".mid")
            
            spectrogram = readAudio(audio_path)
            timesteps = spectrogram.shape[0]
            label = readMidi(midi_path, MIDI_MAPPING, timesteps, 5, vocabulary)

            partitions = timesteps // 400
            data += list(spectrogram.tensor_split(partitions, dim=0))
            labels += list(label.tensor_split(partitions, dim=0))
        data, labels = torch.stack(data), torch.stack(labels)
    
        # Turn them into a Pytorch tensor dataset
        print("\033[96m", "     Creating tensor datasets", "\033[0m", sep="")
        dataset = TensorDataset(data, labels)

        new_path = (path / f"slakh_{split}").with_suffix(".pt")
        print("\033[96m", "     Storing ", "\033[0m", f"{new_path.name}", "\033[96m", " to disk", "\033[0m", sep="")
        torch.save(dataset, new_path)

        print("\033[95m", "     Finished!", "\033[0m", sep="")
    
        # Load dataset and verify that everything is correct
        dataset = torch.load(new_path)
        print("\033[92m", "     Final dataset contains ", "\033[0m", len(dataset), "\033[92m", " entries", "\033[0m", sep="")
        print("\033[92m", "     Each entry has features of shape: ", "\033[0m", dataset[0][0].shape, "\033[92m", ", and labels of shape: ", "\033[0m", dataset[0][1].shape, sep="")
        print("\033[92m", "     Each class has a frequency of: ", "\033[0m", dataset[:][1].round().sum(dim=(0, 1)), sep="")

    # Verify that dataloaders work
    dataloader = DataLoader(torch.load(path / "slakh_train.pt"), batch_size=16)
    num_batches, mean, std = len(dataloader), torch.zeros(1), torch.zeros(1)
    for i, (features, labels) in enumerate(dataloader):
        if i == 0:
            print("\033[92m", "Batched entry in dataloader has features of shape: ", "\033[0m", features.shape, "\033[92m", ", and labels of shape: ", "\033[0m", labels.shape, sep="")
        mean += torch.mean(features, dim=(0, 1, 2))
        std += torch.std(features, dim=(0, 1, 2))

    # Compute mean and std of dataset
    mean /= num_batches
    std /= num_batches
    print("\033[92m", "Training dataset has mean of: ", "\033[0m", mean, "\033[92m", ", and std of: ", "\033[0m", std, sep="")
        
    # At last, print final vocabulary size
    print("\033[95m", "Vocabulary size:", "\033[0m ", len(vocabulary), sep="")