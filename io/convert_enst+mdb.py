import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

from load import readAudio, readAnnotations
from mapping import ENST_MAPPING, MDB_MAPPING


"""
Run this file to turn ENST-Drums + MDB Drums + IDMT-SMT-Drums
into stored PyTorch datasets.

All three datasets use the same thesis frontend:

    Audio
      -> 84-band Mel
      -> Log-Mel or PCEN-Mel
      -> positive first-order temporal difference
      -> 168 features
      -> 400-frame chunks
      -> 3 drum classes

Classes:
    0 = bass drum
    1 = snare drum
    2 = hi-hat
"""


# ============================================================
# Constants
# ============================================================

NUM_LABELS = 3
CHUNK_FRAMES = 400

# IDMT annotations contain onset times in seconds.
# The repository frontend uses 10 ms annotation frames.
SECONDS_PER_FRAME = 0.010

IDMT_MAPPING = {
    "KD": 0,
    "SD": 1,
    "HH": 2,
}


# ============================================================
# Argument parser
# ============================================================

parser = argparse.ArgumentParser(
    "convert_enst+mdb.py"
)

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
    default="MDB Drums",
)

parser.add_argument(
    "--directory_idmt",
    help="The outer directory for the IDMT-SMT-Drums dataset",
    required=False,
    default="IDMT SMT",
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
    help="Process only ENST-Drums and skip MDB and IDMT",
)

parser.add_argument(
    "--enst_mdb_only",
    action="store_true",
    help="Process ENST-Drums and MDB Drums, but skip IDMT",
)

args = parser.parse_args()


# ============================================================
# ENST splits
# ============================================================

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


# ============================================================
# MDB splits
# ============================================================

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


# ============================================================
# IDMT split creation
# ============================================================

def create_idmt_splits(directory_idmt):
    """
    Create deterministic recording-level IDMT train/validation/test
    splits.

    The split is performed before feature extraction so that chunks
    belonging to the same recording can never occur in different
    subsets.

    Current local IDMT package contains 95 MIX recordings.

    Split:
        70% train
        15% validation
        15% test

    A fixed seed makes the split reproducible.
    """

    audio_directory = (
        directory_idmt / "audio"
    )

    recordings = sorted(
        audio_directory.glob("*#MIX.wav")
    )

    if not recordings:
        return [[], [], []]

    rng = random.Random(100)

    recordings = list(recordings)
    rng.shuffle(recordings)

    num_recordings = len(recordings)

    num_train = int(
        round(num_recordings * 0.70)
    )

    num_validation = int(
        round(num_recordings * 0.15)
    )

    train = recordings[:num_train]

    validation_start = num_train
    validation_end = (
        validation_start + num_validation
    )

    validation = recordings[
        validation_start:validation_end
    ]

    test = recordings[
        validation_end:
    ]

    return [
        train,
        validation,
        test,
    ]


# ============================================================
# IDMT XML annotation parser
# ============================================================

def read_idmt_annotations(
    path,
    num_frames,
):
    """
    Read an IDMT XML annotation file.

    IDMT XML annotations contain events such as:

        <event>
            <onsetSec>0.08585</onsetSec>
            <instrument>HH</instrument>
        </event>

    Only KD, SD and HH are retained.

    Returns:
        Tensor with shape [num_frames, 3].
    """

    labels = torch.zeros(
        (num_frames, NUM_LABELS),
        dtype=torch.float32,
    )

    tree = ET.parse(path)
    root = tree.getroot()

    for event in root.findall(".//event"):

        onset_element = event.find(
            "onsetSec"
        )

        instrument_element = event.find(
            "instrument"
        )

        if (
            onset_element is None
            or instrument_element is None
        ):
            continue

        if (
            onset_element.text is None
            or instrument_element.text is None
        ):
            continue

        instrument = (
            instrument_element.text.strip()
        )

        # Ignore TT, CY, OT and any other
        # instruments not part of the thesis.
        if instrument not in IDMT_MAPPING:
            continue

        try:
            onset_seconds = float(
                onset_element.text
            )
        except ValueError:
            continue

        frame = int(
            round(
                onset_seconds
                / SECONDS_PER_FRAME
            )
        )

        if (
            frame < 0
            or frame >= num_frames
        ):
            continue

        label = IDMT_MAPPING[
            instrument
        ]

        labels[frame, label] = 1.0

    return labels


# ============================================================
# Split IDMT recording into 400-frame chunks
# ============================================================

def split_recording(
    spectrogram,
    label,
):
    """
    Split one recording into 400-frame chunks.

    readAudio() pads the feature representation so that the temporal
    dimension is divisible into 400-frame segments.
    """

    timesteps = spectrogram.shape[0]

    if label.shape[0] != timesteps:
        raise RuntimeError(
            "Feature and annotation lengths do not match: "
            f"{timesteps} != {label.shape[0]}"
        )

    if timesteps % CHUNK_FRAMES != 0:
        raise RuntimeError(
            "Feature sequence is not divisible by 400 frames: "
            f"{timesteps}"
        )

    partitions = timesteps // CHUNK_FRAMES

    return (
        list(
            spectrogram.tensor_split(
                partitions,
                dim=0,
            )
        ),
        list(
            label.tensor_split(
                partitions,
                dim=0,
            )
        ),
    )


# ============================================================
# Main conversion
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Declare dataset paths
    # --------------------------------------------------------

    path = (
        Path(__file__).resolve().parent.parent
        / "data"
    )

    output_path = path / "ENST+MDB"

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    enst_path = (
        path / args.directory_enst
    )

    mdb_path = (
        path / args.directory_mdb
    )

    idmt_path = (
        path / args.directory_idmt
    )

    # --------------------------------------------------------
    # Dataset mode
    # --------------------------------------------------------

    if args.enst_only:

        MDB_SPLITS = [
            [],
            [],
            [],
        ]

        IDMT_SPLITS = [
            [],
            [],
            [],
        ]

        dataset_mode = "ENST only"

    elif args.enst_mdb_only:

        IDMT_SPLITS = [
            [],
            [],
            [],
        ]

        dataset_mode = "ENST + MDB"

    else:

        IDMT_SPLITS = create_idmt_splits(
            idmt_path
        )

        dataset_mode = (
            "ENST + MDB + IDMT"
        )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print(
        "\033[96m",
        f"Using representation: "
        f"{args.representation}",
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

    print(
        "\033[96m",
        f"Dataset mode: {dataset_mode}",
        "\033[0m",
        sep="",
    )

    if IDMT_SPLITS[0]:
        print(
            "\033[96m",
            "IDMT recording split: ",
            "\033[0m",
            f"train={len(IDMT_SPLITS[0])}, "
            f"validation={len(IDMT_SPLITS[1])}, "
            f"test={len(IDMT_SPLITS[2])}",
            sep="",
        )

    print(
        "\033[96m",
        "Loading data into lists",
        "\033[0m",
        sep="",
    )

    train_data = []
    train_labels = []

    validation_data = []
    validation_labels = []

    test_data = []
    test_labels = []

    # ========================================================
    # ENST
    # ========================================================

    for drummer in range(3):

        for i, piece in enumerate(
            ENST_SPLITS[drummer]
        ):

            audio_path = (
                enst_path
                / f"drummer_{drummer + 1}"
                / "audio"
                / "wet_mix"
                / piece
            ).with_suffix(".wav")

            accompaniment_path = (
                enst_path
                / f"drummer_{drummer + 1}"
                / "audio"
                / "accompaniment"
                / piece
            ).with_suffix(".wav")

            annotation_path = (
                enst_path
                / f"drummer_{drummer + 1}"
                / "annotation"
                / piece
            ).with_suffix(".txt")

            # Some historical ENST split entries are
            # not present in the current package.
            if not (
                audio_path.is_file()
                and accompaniment_path.is_file()
                and annotation_path.is_file()
            ):

                print(
                    "\033[93m",
                    "Skipping missing ENST recording: "
                    f"{piece} "
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

            timesteps = (
                spectrogram.shape[0]
            )

            label = readAnnotations(
                annotation_path,
                ENST_MAPPING,
                timesteps,
                NUM_LABELS,
            )

            data_chunks, label_chunks = (
                split_recording(
                    spectrogram,
                    label,
                )
            )

            if drummer < 2:

                train_data += data_chunks
                train_labels += label_chunks

            elif (
                i
                < len(
                    ENST_SPLITS[drummer]
                ) // 2
            ):

                validation_data += data_chunks
                validation_labels += label_chunks

            else:

                test_data += data_chunks
                test_labels += label_chunks

    # ========================================================
    # MDB
    # ========================================================

    for drummer in range(3):

        for i, piece in enumerate(
            MDB_SPLITS[drummer]
        ):

            # Use the full musical mix rather than
            # the isolated drum-only recording.
            audio_path = (
                mdb_path
                / "audio"
                / "full_mix"
                / f"{piece}_MIX"
            ).with_suffix(".wav")

            # MDB class annotations contain:
            #
            # KD = kick
            # SD = snare
            # HH = hi-hat
            # TT = toms
            # CY = cymbals
            # OT = other
            #
            # Only KD/SD/HH are retained.
            annotation_path = (
                mdb_path
                / "annotations"
                / "class"
                / f"{piece}_class"
            ).with_suffix(".txt")

            if not audio_path.is_file():

                print(
                    "\033[93m",
                    f"Skipping missing MDB audio: "
                    f"{piece}",
                    "\033[0m",
                    sep="",
                )

                continue

            if not annotation_path.is_file():

                print(
                    "\033[93m",
                    f"Skipping missing MDB annotation: "
                    f"{piece}",
                    "\033[0m",
                    sep="",
                )

                continue

            spectrogram = readAudio(
                audio_path,
                representation=args.representation,
            )

            timesteps = (
                spectrogram.shape[0]
            )

            label = readAnnotations(
                annotation_path,
                MDB_MAPPING,
                timesteps,
                NUM_LABELS,
            )

            data_chunks, label_chunks = (
                split_recording(
                    spectrogram,
                    label,
                )
            )

            if drummer < 2:

                train_data += data_chunks
                train_labels += label_chunks

            elif (
                i
                < len(
                    MDB_SPLITS[drummer]
                ) // 2
            ):

                validation_data += data_chunks
                validation_labels += label_chunks

            else:

                test_data += data_chunks
                test_labels += label_chunks

    # ========================================================
    # IDMT-SMT-Drums
    # ========================================================

    for split_index in range(3):

        split_recordings = (
            IDMT_SPLITS[split_index]
        )

        for recording_index, audio_path in enumerate(
            split_recordings,
            start=1,
        ):

            annotation_path = (
                idmt_path
                / "annotation_xml"
                / f"{audio_path.stem}.xml"
            )

            if not audio_path.is_file():

                print(
                    "\033[93m",
                    "Skipping missing IDMT audio: "
                    f"{audio_path.name}",
                    "\033[0m",
                    sep="",
                )

                continue

            if not annotation_path.is_file():

                print(
                    "\033[93m",
                    "Skipping missing IDMT annotation: "
                    f"{annotation_path.name}",
                    "\033[0m",
                    sep="",
                )

                continue

            print(
                "\033[96m",
                f"IDMT "
                f"{['train', 'validation', 'test'][split_index]} "
                f"[{recording_index}/"
                f"{len(split_recordings)}]: "
                f"{audio_path.name}",
                "\033[0m",
                sep="",
            )

            spectrogram = readAudio(
                audio_path,
                representation=args.representation,
            )

            timesteps = (
                spectrogram.shape[0]
            )

            label = read_idmt_annotations(
                annotation_path,
                timesteps,
            )

            data_chunks, label_chunks = (
                split_recording(
                    spectrogram,
                    label,
                )
            )

            if split_index == 0:

                train_data += data_chunks
                train_labels += label_chunks

            elif split_index == 1:

                validation_data += data_chunks
                validation_labels += label_chunks

            else:

                test_data += data_chunks
                test_labels += label_chunks

    # ========================================================
    # Convert lists to tensors
    # ========================================================

    print(
        "\033[96m",
        "Creating tensor datasets",
        "\033[0m",
        sep="",
    )

    if not train_data:
        raise RuntimeError(
            "Training dataset is empty."
        )

    if not validation_data:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    if not test_data:
        raise RuntimeError(
            "Test dataset is empty."
        )

    train_data = torch.stack(
        train_data
    )

    train_labels = torch.stack(
        train_labels
    )

    validation_data = torch.stack(
        validation_data
    )

    validation_labels = torch.stack(
        validation_labels
    )

    test_data = torch.stack(
        test_data
    )

    test_labels = torch.stack(
        test_labels
    )

    # ========================================================
    # PyTorch datasets
    # ========================================================

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

    # ========================================================
    # Verify split sizes
    # ========================================================

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

    # ========================================================
    # Store every split
    # ========================================================

    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }

    for split, dataset in datasets.items():

        new_path = (
            output_path
            / (
                f"enst+mdb+idmt_"
                f"{args.representation}_"
                f"{split}"
            )
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

        torch.save(
            dataset,
            new_path,
        )

        print(
            "\033[95m",
            "     Finished!",
            "\033[0m",
            sep="",
        )

        # ----------------------------------------------------
        # Load dataset and verify
        # ----------------------------------------------------

        dataset_check = torch.load(
            new_path,
            weights_only=False,
        )

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
            dataset_check[:][1]
            .round()
            .sum(dim=(0, 1)),
            sep="",
        )

    # ========================================================
    # Verify DataLoader
    # ========================================================

    dataloader = DataLoader(
        train_dataset,
        batch_size=16,
    )

    num_batches = len(
        dataloader
    )

    mean = torch.zeros(1)
    std = torch.zeros(1)

    for i, (features, labels) in enumerate(
        dataloader
    ):

        if i == 0:

            print(
                "\033[92m",
                "Batched entry in dataloader has "
                "features of shape: ",
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

    mean /= num_batches
    std /= num_batches

    print(
        "\033[92m",
        "Training dataset has mean of: ",
        "\033[0m",
        mean,
        "\033[92m",
        ", and std of: ",
        "\033[0m",
        std,
        sep="",
    )

