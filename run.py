import os
import tempfile
from pathlib import Path

# Keep Ray/Tune temporary files on the E: drive when running locally.
# On Kaggle/Linux, use the default temporary directory.
if os.name == "nt":

    TEMP_DIR = Path("E:/temp")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["TEMP"] = str(TEMP_DIR)
    os.environ["TMP"] = str(TEMP_DIR)

    tempfile.tempdir = str(TEMP_DIR)


import argparse
import json
import random

import numpy as np
import ray
from ray.tune import Checkpoint
import torch

from models import CNN
from preprocess import create_transform, load_dataset, MelRecordingDataset
from models.trainable_pcen import TrainablePCEN
from load import trainable_pcen_features
from train import prepare_features_for_model, train_model
from evaluate import evaluate_model


# ----------------------------------------------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=[
            "enst+mdb",
            "enst+mdb+idmt",
            "enst_idmt_polyphonic",
            "egmd",
            "slakh",
            "adtof_yt",
        ],
        default=["enst+mdb+idmt"],
        help="Dataset(s) to use.",
    )

    parser.add_argument(
        "--representation",
        choices=[
            "logmel",
            "pcen",
            "trainable_pcen",
            "optimized_pcen",
        ],
        default="logmel",
        help="Acoustic representation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=15,
        help="Number of Ray Tune samples.",
    )

    parser.add_argument(
        "--evaluate-checkpoint",
        type=str,
        default=None,
        help="Path to an existing Ray checkpoint to evaluate without training.",
    )

    parser.add_argument(
        "--early-stop",
        type=int,
        default=15,
        help="Number of epochs without improvement.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # ----------------------------------------------------------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------------------------------------------------------

    repo_dir = Path(__file__).resolve().parent

    data_dir = (
        repo_dir
        / "data"
    )

    dataset_directories = {

        "enst+mdb":
            data_dir
            / "ENST+MDB",

        "enst+mdb+idmt":
            data_dir
            / "ENST+MDB",

        "enst_idmt_polyphonic":
            data_dir
            / "ENST_IDMT_Polyphonic",

        "egmd":
            data_dir
            / "e-gmd-v1.0.0",

        "slakh":
            data_dir
            / "slakh2100_flac_redux",

        "adtof_yt":
            data_dir
            / "adtof",
    }

    dataset_paths = [
        dataset_directories[dataset]
        for dataset in args.dataset
    ]

    # ----------------------------------------------------------------------------------------------------------------
    # Dataset filenames
    # ----------------------------------------------------------------------------------------------------------------

    if args.representation == "trainable_pcen":

        if args.dataset != ["enst_idmt_polyphonic"]:
            parser.error(
                "trainable_pcen currently requires "
                "--dataset enst_idmt_polyphonic"
            )

        # MelRecordingDataset expects the root containing
        # mel_recordings/train, validation, and test.
        root = dataset_paths[0]

        train_paths = [root]
        val_paths = [root]
        test_paths = [root]

    else:

        suffix = f"_{args.representation}"

        train_paths = [
            dataset_path / f"{dataset}{suffix}_train.pt"
            for dataset_path, dataset
            in zip(dataset_paths, args.dataset)
        ]

        val_paths = [
            dataset_path / f"{dataset}{suffix}_validation.pt"
            for dataset_path, dataset
            in zip(dataset_paths, args.dataset)
        ]

        test_paths = [
            dataset_path / f"{dataset}{suffix}_test.pt"
            for dataset_path, dataset
            in zip(dataset_paths, args.dataset)
        ]

    # ----------------------------------------------------------------------------------------------------------------
    # Verify dataset files
    # ----------------------------------------------------------------------------------------------------------------

    print(
        "Dataset files:"
    )

    for path in train_paths:

        print(
            f"  Train: {path}"
        )

    for path in val_paths:

        print(
            f"  Validation: {path}"
        )

    for path in test_paths:

        print(
            f"  Test: {path}"
        )

    missing_files = [
        path
        for path in (
            train_paths
            + val_paths
            + test_paths
        )
        if not path.exists()
    ]

    if missing_files:

        print(
            "\nMissing dataset files:"
        )

        for path in missing_files:

            print(
                f"  {path}"
            )

        raise FileNotFoundError(
            "One or more required dataset files are missing."
        )

    # ----------------------------------------------------------------------------------------------------------------
    # Study
    # ----------------------------------------------------------------------------------------------------------------

    study = (
        "Architecture"
        if len(args.dataset) == 1
        else "Dataset"
    )

    experiment = (
        f"{args.dataset[0]}_"
        f"{args.representation}"
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Device
    # ----------------------------------------------------------------------------------------------------------------

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(
        f"\nUsing device: {device}"
    )

    print(
        f"Representation: "
        f"{args.representation}"
    )

    print(
        f"Dataset: "
        f"{args.dataset}"
    )

    print(
        f"Study: {study}"
    )

    print(
        f"Experiment: {experiment}"
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Load training datasets and calculate training-only normalization
    # ----------------------------------------------------------------------------------------------------------------

    if args.representation == "trainable_pcen":

        train_datasets = [
            MelRecordingDataset(path, split="train")
            for path in train_paths
        ]

    else:

        train_datasets = [
            load_dataset(path)
            for path in train_paths
        ]

    train_dataset = torch.utils.data.ConcatDataset(
        train_datasets
    )

    print(
        f"\\nTraining dataset size: {len(train_dataset)}"
    )

    print("Computing training-only normalization statistics...")

    if args.representation == "trainable_pcen":

        # Statistics are calculated once using initial PCEN parameters.
        # They remain fixed during training, avoiding validation/test leakage.
        initial_pcen = TrainablePCEN().cpu().eval()

        total_sum = torch.zeros((), dtype=torch.float64)
        total_square_sum = torch.zeros((), dtype=torch.float64)
        total_values = 0

        with torch.no_grad():

            for index, (mel, _) in enumerate(train_dataset):

                features = trainable_pcen_features(
                    mel,
                    initial_pcen,
                )

                values = features.to(torch.float64)

                total_sum += values.sum()
                total_square_sum += values.square().sum()
                total_values += values.numel()

                if (index + 1) % 25 == 0:
                    print(
                        f"  Processed {index + 1}/"
                        f"{len(train_dataset)} recordings"
                    )

        if total_values == 0:
            raise RuntimeError("No training features found.")

        mean64 = total_sum / total_values

        variance64 = (
            total_square_sum / total_values - mean64.square()
        ).clamp(min=0.0)

        mean = mean64.float()
        std = variance64.sqrt().clamp(min=1e-8).float()

    else:

        # Compute training-only statistics without concatenating
        # hundreds of megabytes of precomputed features.
        print(
            "Calculating chunk-based normalization...",
            flush=True,
        )

        total_sum = torch.zeros((), dtype=torch.float64)
        total_squared = torch.zeros((), dtype=torch.float64)
        total_count = 0
        completed = 0

        with torch.no_grad():
            for dataset in train_datasets:
                for features, _ in dataset:
                    values = features.to(torch.float64)

                    total_sum += values.sum()
                    total_squared += values.square().sum()
                    total_count += values.numel()
                    completed += 1

                    if completed % 200 == 0:
                        print(
                            f"  Normalized statistics: "
                            f"{completed}/{len(train_dataset)} chunks",
                            flush=True,
                        )

        if total_count < 2:
            raise RuntimeError(
                "Not enough training values for normalization."
            )

        mean64 = total_sum / total_count

        # Match torch.std() default: unbiased sample variance.
        variance64 = (
            total_squared - total_count * mean64.square()
        ) / (total_count - 1)

        mean = mean64.float()
        std = variance64.clamp(min=1e-16).sqrt().float()

        print(
            f"Normalization completed for {completed} chunks.",
            flush=True,
        )

    print(f"Training mean: {mean}")
    print(f"Training std: {std}")

    # ----------------------------------------------------------------------------------------------------------------
    # Ray configuration
    # ----------------------------------------------------------------------------------------------------------------

    config = {

        "train_paths":
            train_paths,

        "val_paths":
            val_paths,

        "mean":
            mean,

        "std":
            std,

        "optimizer":
            "AdamW",

        "Model":
            CNN,

        "parameters":
            {},

        "lr":
            1e-3,

        "weight_decay":
            1e-4,

        "device":
            str(device),

        "seed":
            args.seed,

        "representation":
            args.representation,

        "num_workers":
            0,

        "early_stop":
            args.early_stop,

        "num_epochs":
            args.num_epochs,

        "batch_size":
            args.batch_size,
    }

    # ----------------------------------------------------------------------------------------------------------------
    # Training OR existing checkpoint
    # ----------------------------------------------------------------------------------------------------------------

    ray_initialized = False

    if args.evaluate_checkpoint is not None:

        checkpoint_path = Path(
            args.evaluate_checkpoint
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                "The specified checkpoint does not exist:\n"
                f"{checkpoint_path}"
            )

        model_path = (
            checkpoint_path
            / "model.pt"
        )

        if not model_path.exists():

            raise FileNotFoundError(
                "The checkpoint does not contain model.pt:\n"
                f"{model_path}"
            )

        print(
            "\nEvaluation-only mode."
        )

        print(
            f"Using existing checkpoint: "
            f"{checkpoint_path}"
        )

        best_checkpoint = (
            Checkpoint.from_directory(
                checkpoint_path.as_posix()
            )
        )

    else:

        print(
            "\nInitializing Ray..."
        )

        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=2,
            num_gpus=(
                1
                if torch.cuda.is_available()
                else 0
            ),
        )

        ray_initialized = True

        # ------------------------------------------------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------------------------------------------------

        print(
            "\nStarting training..."
        )

        results = train_model(
            config=config,
            num_samples=args.num_samples,
        )

        if results is None:

            raise RuntimeError(
                "Training did not return any results."
            )

        dataframe = results.get_dataframe()

        if dataframe.empty:

            raise RuntimeError(
                "Ray Tune returned no completed trial results."
            )

        # ------------------------------------------------------------------------------------------------------------
        # Best trial
        # ------------------------------------------------------------------------------------------------------------

        best_result = (
            results.get_best_result(
                metric="Micro F1",
                mode="max",
            )
        )

        if best_result is None:

            raise RuntimeError(
                "No successful training trial was found."
            )

        best_micro_f1 = (
            best_result.metrics.get(
                "Micro F1"
            )
        )

        print(
            "\nTraining completed."
        )

        print(
            f"Best validation Micro F1: "
            f"{best_micro_f1}"
        )

        # ------------------------------------------------------------------------------------------------------------
        # Best checkpoint
        # ------------------------------------------------------------------------------------------------------------

        best_checkpoint = (
            best_result.checkpoint
        )

        if best_checkpoint is None:

            raise RuntimeError(
                "The best training trial did not "
                "produce a checkpoint."
            )

        print(
            f"Best checkpoint: "
            f"{best_checkpoint}"
        )

    # ----------------------------------------------------------------------------------------------------------------
    # Test dataset
    # ----------------------------------------------------------------------------------------------------------------

    use_trainable_pcen = (
        args.representation == "trainable_pcen"
    )

    if use_trainable_pcen:
        test_datasets = [
            MelRecordingDataset(root, split="test")
            for root in test_paths
        ]
    else:
        test_datasets = [
            load_dataset(path)
            for path in test_paths
        ]

    test_dataset = torch.utils.data.ConcatDataset(
        test_datasets
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1 if use_trainable_pcen else args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Test transform and model
    # ----------------------------------------------------------------------------------------------------------------

    test_transform = create_transform(
        mean=mean,
        std=std,
        channels_last=False,
    )

    Model = config["Model"]
    model_parameters = config.get("parameters", {})

    test_model = Model(**model_parameters)

    with best_checkpoint.as_directory() as checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)

        state_dict = torch.load(
            checkpoint_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )

        test_model.load_state_dict(state_dict)

        if use_trainable_pcen:
            pcen_path = checkpoint_dir / "pcen.pt"

            if not pcen_path.is_file():
                raise FileNotFoundError(
                    f"Learnable PCEN checkpoint missing: {pcen_path}"
                )

            learned_pcen = TrainablePCEN()
            learned_pcen.load_state_dict(
                torch.load(
                    pcen_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )

    if use_trainable_pcen:

        class PCENEvaluationModel(torch.nn.Module):
            """Evaluate PCEN on full recordings, then use CNN chunks."""

            def __init__(self, cnn, pcen, normalizer):
                super().__init__()
                self.cnn = cnn
                self.pcen = pcen
                self.normalizer = normalizer

            def forward(self, linear_mel):
                # Input: [1, recording_frames, 84, 1]
                if linear_mel.ndim != 4 or linear_mel.shape[0] != 1:
                    raise ValueError(
                        "Learnable PCEN evaluation expects one recording."
                    )

                features = trainable_pcen_features(
                    linear_mel.squeeze(0),
                    self.pcen,
                )

                features = self.normalizer(features)
                cnn_inputs = prepare_features_for_model(features)
                outputs = self.cnn(cnn_inputs)

                # Reassemble chunks so onset matching sees a
                # continuous recording: [1, recording_frames, 3].
                return outputs.reshape(1, -1, outputs.shape[-1])

        evaluation_model = PCENEvaluationModel(
            test_model,
            learned_pcen,
            test_transform,
        )

        def test_model_transform(inputs):
            # The wrapper performs PCEN and normalization.
            return inputs

    else:

        evaluation_model = test_model

        def test_model_transform(inputs):
            # Normalize baseline test data just as in training.
            inputs = test_transform(inputs)
            return prepare_features_for_model(inputs)

    # ----------------------------------------------------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------------------------------------------------

    print("\\nEvaluating on test set...")

    with torch.no_grad():
        test_metrics = evaluate_model(
            evaluation_model,
            test_loader,
            test_model_transform,
            device=device,
        )

    # ----------------------------------------------------------------------------------------------------------------
    # Print test results
    # ----------------------------------------------------------------------------------------------------------------

    print(
        "\nTest results:"
    )

    print(
        json.dumps(
            test_metrics,
            indent=2,
            default=str,
        )
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Save experiment information
    # ----------------------------------------------------------------------------------------------------------------

    output_dir = (
        repo_dir
        / "experiments"
        / args.representation
        / study
        / experiment
        / args.dataset[0].upper()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_dir
        / "test_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            test_metrics,
            f,
            indent=2,
            default=str,
        )

    print(
        f"\nResults saved to: "
        f"{results_path}"
    )

    if ray_initialized:

        ray.shutdown()


if __name__ == "__main__":

    main()
