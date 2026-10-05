import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import argparse

from load import readAudio, readAnnotations
from mapping import ENST_MAPPING, MDB_MAPPING


"""Run this file to turn ENST-Drums + MDB Drums into stored PyTorch datasets."""


# Number of thesis drum classes:
# 0 = bass drum
# 1 = snare drum
# 2 = hi-hat
NUM_LABELS = 3


# Declare an argument parser for this file
parser = argparse.ArgumentParser("convert_enst+mdb.py")

parser.add_argument(
    "--directory_enst",
    help="The outer directory for the ENST-Drums dataset",
    required=False,
    default="ENST-drums-public",
)

parser.add_argument(
    "--directory_mdb",
    help="The outer directory for the MDB Drums dataset",
    required=False,
    default="MDBDrums-master/MDB Drums",
)

parser.add_argument(
    "--representation",
    help="Acoustic representation to use",
    choices=["logmel", "pcen"],
    required=False,
    default="logmel",
)

parser.add_argument(
    "--enst_only",
    action="store_true",
    help="Process only ENST-Drums and skip MDB Drums",
)

args = parser.parse_args()


# Splits from ADTOF-github
# Originally from Vogl et al.
ENST_SPLITS = [
    [
        "107_minus-one_salsa_sticks",
        "108_minus-one_rock-60s_sticks",
        "109_minus-one_metal_sticks",
        "110_minus-one_musette_brushes",
        "111_minus-one_funky_rods",
        "112_minus-one_funk_rods",
        "113_minus-one_charleston_sticks",
        "114_minus-one_celtic-rock_brushes",
        "115_minus-one_bossa_brushes",
        "121_MIDI-minus-one_bigband_brushes",
        "123_MIDI-minus-one_blues-102_sticks",
        "125_MIDI-minus-one_country-120_sticks",
        "127_MIDI-minus-one_disco-108_sticks",
        "129_MIDI-minus-one_funk-101_sticks",
        "131_MIDI-minus-one_grunge_sticks",
        "133_MIDI-minus-one_nu-soul_sticks",
        "135_MIDI-minus-one_rock-113_sticks",
        "137_MIDI-minus-one_rock'n'roll-188_sticks",
        "139_MIDI-minus-one_soul-120-marvin-gaye_sticks",
        "141_MIDI-minus-one_soul-98_sticks",
        "143_MIDI-minus-one_fusion-125_sticks",
    ],
    [
        "115_minus-one_salsa_sticks",
        "116_minus-one_rock-60s_sticks",
        "117_minus-one_metal_sticks",
        "118_minus-one_musette_brushes",
        "119_minus-one_funky_sticks",
        "120_minus-one_funk_sticks",
        "121_minus-one_charleston_sticks",
        "122_minus-one_celtic-rock_sticks",
        "123_minus-one_celtic-rock-better-take_sticks",
        "124_minus-one_bossa_sticks",
        "130_MIDI-minus-one_bigband_sticks",
        "132_MIDI-minus-one_blues-102_sticks",
        "134_MIDI-minus-one_country-120_sticks",
        "136_MIDI-minus-one_disco-108_sticks",
        "138_MIDI-minus-one_funk-101_sticks",
        "140_MIDI-minus-one_grunge_sticks",
        "142_MIDI-minus-one_nu-soul_sticks",
        "144_MIDI-minus-one_rock-113_sticks",
        "146_MIDI-minus-one_rock'n'roll-188_sticks",
        "148_MIDI-minus-one_soul-120-marvin-gaye_sticks",
        "150_MIDI-minus-one_soul-98_sticks",
        "152_MIDI-minus-one_fusion-125_sticks",
    ],
    [
        "126_minus-one_salsa_sticks",
        "127_minus-one_rock-60s_sticks",
        "128_minus-one_metal_sticks",
        "129_minus-one_musette_sticks",
        "130_minus-one_funky_sticks",
        "131_minus-one_funk_sticks",
        "132_minus-one_charleston_sticks",
        "133_minus-one_celtic-rock_sticks",
        "134_minus-one_bossa_sticks",
        "140_MIDI-minus-one_bigband_sticks",
        "142_MIDI-minus-one_blues-102_sticks",
        "144_MIDI-minus-one_country-120_sticks",
        "146_MIDI-minus-one_disco-108_sticks",
        "148_MIDI-minus-one_funk-101_sticks",
        "150_MIDI-minus-one_grunge_sticks",
        "152_MIDI-minus-one_grunge_sticks",
        "154_MIDI-minus-one_fusion-125_sticks",
        "156_MIDI-minus-one_rock-113_sticks",
        "158_MIDI-minus-one_rock'n'roll-188_sticks",
        "160_MIDI-minus-one_soul-98_sticks",
        "162_MIDI-minus-one_fusion-125_sticks",
    ],
]


MDB_SPLITS = [
    [
        "MusicDelta_Punk",
        "MusicDelta_CoolJazz",
        "MusicDelta_Disco",
        "MusicDelta_SwingJazz",
        "MusicDelta_Rockabilly",
        "MusicDelta_Gospel",
        "MusicDelta_BebopJazz",
    ],
    [
        "MusicDelta_FunkJazz",
        "MusicDelta_FreeJazz",
        "MusicDelta_Reggae",
        "MusicDelta_LatinJazz",
        "MusicDelta_Britpop",
        "MusicDelta_FusionJazz",
        "MusicDelta_Shadows",
        "MusicDelta_80sRock",
    ],
    [
        "MusicDelta_Beatles",
        "MusicDelta_Grunge",
        "MusicDelta_Zeppelin",
        "MusicDelta_ModalJazz",
        "MusicDelta_Country1",
        "MusicDelta_SpeedMetal",
        "MusicDelta_Rock",
        "MusicDelta_Hendrix",
    ],
]


# Skip MDB Drums when testing with ENST only.
if args.enst_only:
    MDB_SPLITS = [[], [], []]


# Set a seed for predictable splitting
seed = 100


if __name__ == "__main__":

    # Declare the path to the dataset directory
    # Declare the path to the dataset directory
    path = Path(__file__).resolve().parent.parent / "data" / "ENST+MDB"

    # Create the output directory if it does not exist.
    path.mkdir(parents=True, exist_ok=True)
    print(
        "\033[96m",
        f"Using representation: {args.representation}",
        "\033[0m",
        sep="",
    )

    print(
        "\033[96m",
        "Number of labels: ",
        "\033[0m",
        NUM_LABELS,
        sep="",
    )

    if args.enst_only:
        print(
            "\033[96m",
            "Dataset mode: ENST only",
            "\033[0m",
            sep="",
        )

    print(
        "\033[96m",
        "Loading data into lists",
        "\033[0m",
        sep="",
    )

    train_data, train_labels = [], []
    validation_data, validation_labels = [], []
    test_data, test_labels = [], []

    # ============================================================
    # ENST
    # ============================================================

    for drummer in range(3):

        for i, piece in enumerate(ENST_SPLITS[drummer]):

            audio_path = (
                path
                / args.directory_enst
                / f"drummer_{drummer + 1}"
                / "audio"
                / "wet_mix"
                / piece
            ).with_suffix(".wav")

            accompaniment_path = (
                path
                / args.directory_enst
                / f"drummer_{drummer + 1}"
                / "audio"
                / "accompaniment"
                / piece
            ).with_suffix(".wav")

            annotation_path = (
                path
                / args.directory_enst
                / f"drummer_{drummer + 1}"
                / "annotation"
                / piece
            ).with_suffix(".txt")

            # Some entries in the historical ENST split definition
            # are not present in the current ENST-Drums package.
            # Do not substitute different recordings: skip them.
            if not (
                audio_path.is_file()
                and accompaniment_path.is_file()
                and annotation_path.is_file()
            ):
                print(
                    "\033[93m",
                    f"Skipping missing ENST recording: {piece} "
                    f"(drummer_{drummer + 1})",
                    "\033[0m",
                    sep="",
                )
                continue

            spectrogram = readAudio(
                audio_path,
                accompaniment_path,
                representation=args.representation,
            )

            timesteps = spectrogram.shape[0]

            label = readAnnotations(
                annotation_path,
                ENST_MAPPING,
                timesteps,
                NUM_LABELS,
            )

            partitions = timesteps // 400

            if drummer < 2:

                train_data += list(
                    spectrogram.tensor_split(partitions, dim=0)
                )

                train_labels += list(
                    label.tensor_split(partitions, dim=0)
                )

            elif i < len(ENST_SPLITS[drummer]) // 2:

                validation_data += list(
                    spectrogram.tensor_split(partitions, dim=0)
                )

                validation_labels += list(
                    label.tensor_split(partitions, dim=0)
                )

            else:

                test_data += list(
                    spectrogram.tensor_split(partitions, dim=0)
                )

                test_labels += list(
                    label.tensor_split(partitions, dim=0)
                )

    # ============================================================
    # MDB
    # ============================================================

        for i, piece in enumerate(MDB_SPLITS[drummer]):

            audio_path = (
                path
                / args.directory_mdb
                / "audio"
                / "full_mix"
                / f"{piece}_MIX"
            ).with_suffix(".wav")

            annotation_path = (
                path
                / args.directory_mdb
                / "annotations"
                / "subclass"
                / f"{piece}_subclass"
            ).with_suffix(".txt")

            spectrogram = readAudio(
                audio_path,
                representation=args.representation,
            )

            timesteps = spectrogram.shape[0]

            label = readAnnotations(
                annotation_path,
                MDB_MAPPING,
                timesteps,
                NUM_LABELS,
            )

            partitions = timesteps // 400

            if drummer < 2:

                train_data += list(
                    spectrogram.tensor_split(partitions, dim=0)
                )

                train_labels += list(
                    label.tensor_split(partitions, dim=0)
                )

            elif i < len(MDB_SPLITS[drummer]) // 2:

                validation_data += list(
                    spectrogram.tensor_split(partitions, dim=0)
                )

                validation_labels += list(
                    label.tensor_split(partitions, dim=0)
                )

            else:

                test_data += list(
                    spectrogram.tensor_split(partitions, dim=0)
                )

                test_labels += list(
                    label.tensor_split(partitions, dim=0)
                )

    # Convert lists to tensors
    train_data = torch.stack(train_data)
    train_labels = torch.stack(train_labels)

    validation_data = torch.stack(validation_data)
    validation_labels = torch.stack(validation_labels)

    test_data = torch.stack(test_data)
    test_labels = torch.stack(test_labels)

    # Turn them into PyTorch tensor datasets
    print(
        "\033[96m",
        "Creating tensor datasets",
        "\033[0m",
        sep="",
    )

    train_dataset = TensorDataset(
        train_data,
        train_labels,
    )

    validation_dataset = TensorDataset(
        validation_data,
        validation_labels,
    )

    test_dataset = TensorDataset(
        test_data,
        test_labels,
    )

    # Verify split sizes
    print(
        "\033[96m",
        "Train size: ",
        "\033[0m",
        len(train_dataset),
        sep="",
    )

    print(
        "\033[96m",
        "Validation size: ",
        "\033[0m",
        len(validation_dataset),
        sep="",
    )

    print(
        "\033[96m",
        "Test size: ",
        "\033[0m",
        len(test_dataset),
        sep="",
    )

    # Store every split
    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }

    for split, dataset in datasets.items():

        new_path = (
            path
            / f"enst+mdb_{args.representation}_{split}"
        ).with_suffix(".pt")

        print(
            "\033[96m",
            "     Storing ",
            "\033[0m",
            new_path.name,
            "\033[96m",
            " to disk",
            "\033[0m",
            sep="",
        )

        torch.save(dataset, new_path)

        print(
            "\033[95m",
            "     Finished!",
            "\033[0m",
            sep="",
        )

        # Load dataset and verify
        dataset_check = torch.load(new_path, weights_only=False)

        print(
            "\033[92m",
            "     Final dataset contains ",
            "\033[0m",
            len(dataset_check),
            "\033[92m",
            " entries",
            "\033[0m",
            sep="",
        )

        print(
            "\033[92m",
            "     Each entry has features of shape: ",
            "\033[0m",
            dataset_check[0][0].shape,
            "\033[92m",
            ", and labels of shape: ",
            "\033[0m",
            dataset_check[0][1].shape,
            sep="",
        )

        print(
            "\033[92m",
            "     Each class has a frequency of: ",
            "\033[0m",
            dataset_check[:][1].round().sum(dim=(0, 1)),
            sep="",
        )

    # Verify that dataloaders work
    dataloader = DataLoader(
        train_dataset,
        batch_size=16,
    )

    num_batches = len(dataloader)
    mean = torch.zeros(1)
    std = torch.zeros(1)

    for i, (features, labels) in enumerate(dataloader):

        if i == 0:
            print(
                "\033[92m",
                "Batched entry in dataloader has features of shape: ",
                "\033[0m",
                features.shape,
                "\033[92m",
                ", and labels shape: ",
                "\033[0m",
                labels.shape,
                sep="",
            )

        mean += torch.mean(
            features,
            dim=(0, 1, 2),
        )

        std += torch.std(
            features,
            dim=(0, 1, 2),
        )

    # Compute mean and std of dataset
    mean /= num_batches
    std /= num_batches

    print(
        "\033[92m",
        "Training dataset has mean of: ",
        "\033[0m",
        mean,
        "\033[92m",
        ", and std of: ",
        std,
        sep="",
    )

