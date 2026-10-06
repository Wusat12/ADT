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
from preprocess import create_transform, load_dataset
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

    suffix = (
        f"_{args.representation}"
    )

    train_paths = [
        dataset_path
        / f"{dataset}{suffix}_train.pt"
        for dataset_path, dataset
        in zip(
            dataset_paths,
            args.dataset,
        )
    ]

    val_paths = [
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
    # Load training datasets
    # ----------------------------------------------------------------------------------------------------------------

    train_datasets = [
        load_dataset(path)
        for path in train_paths
    ]

    train_dataset = torch.utils.data.ConcatDataset(
        train_datasets
    )

    print(
        f"\nTraining dataset size: "
        f"{len(train_dataset)}"
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Compute normalization statistics
    # ----------------------------------------------------------------------------------------------------------------

    print(
        "Computing normalization statistics..."
    )

    all_features = []

    for dataset in train_datasets:

        for features, _ in dataset:

            all_features.append(
                features
            )

    features = torch.cat(
        all_features,
        dim=0,
    )

    mean = features.mean()
    std = features.std()

    print(
        f"Training mean: {mean}"
    )

    print(
        f"Training std: {std}"
    )

    del all_features
    del features

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
            num_cpus=1,
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

    test_datasets = [
        load_dataset(path)
        for path in test_paths
    ]

    test_dataset = torch.utils.data.ConcatDataset(
        test_datasets
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Test transform
    # ----------------------------------------------------------------------------------------------------------------

    test_transform = create_transform(
        mean=mean,
        std=std,
        channels_last=False,
    )

    for dataset in test_datasets:

        dataset.transform = (
            test_transform
        )

    # ----------------------------------------------------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------------------------------------------------

    print(
        "\nEvaluating on test set..."
    )

    Model = config["Model"]

    model_parameters = config.get(
        "parameters",
        {},
    )

    test_model = Model(
        **model_parameters
    )

    with best_checkpoint.as_directory() as checkpoint_dir:

        model_path = (
            Path(checkpoint_dir)
            / "model.pt"
        )

        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )

    test_model.load_state_dict(
        state_dict
    )

    def test_model_transform(inputs):

        inputs = prepare_features_for_model(
            inputs
        )

        return inputs

    test_metrics = evaluate_model(
        test_model,
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
