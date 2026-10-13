import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

import ray
from ray import train, tune

from preprocess import (
    compute_infrequency_weights,
    create_transform,
    load_dataset,
    MelRecordingDataset,
)

from models.trainable_pcen import TrainablePCEN
from load import trainable_pcen_features

from evaluate import (
    compute_peaks,
    compute_predictions,
    f_measure,
)


# ----------------------------------------------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------------------------
# Feature layout
# ----------------------------------------------------------------------------------------------------------------

def prepare_features_for_model(features):
    """
    Convert dataset features from:

        [B, T, F, C]

    to CNN format:

        [B, C, T, F]

    The converted datasets contain features with shape:

        [T, 168, 1]

    Therefore the DataLoader produces:

        [B, T, 168, 1]
    """

    if features.ndim != 4:
        raise RuntimeError(
            "Expected a 4D feature tensor, "
            f"but received shape {tuple(features.shape)}."
        )

    # Dataset format:
    # [B, T, F, C]
    if features.shape[-1] == 1:

        features = features.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

    # Already in CNN format:
    # [B, C, T, F]
    elif features.shape[1] == 1:

        features = features.contiguous()

    else:

        raise RuntimeError(
            "Unable to determine feature layout. "
            f"Received tensor shape {tuple(features.shape)}."
        )

    return features


# ----------------------------------------------------------------------------------------------------------------
# Single Ray Tune trial
# ----------------------------------------------------------------------------------------------------------------

def _train_trial(config):
    """
    Execute one training trial.

    This function is called by Ray Tune.
    """

    seed = config.get(
        "seed",
        42,
    )

    set_seed(seed)

    # ----------------------------------------------------------------------------------------------------------------
    # Device
    # ----------------------------------------------------------------------------------------------------------------

    device_name = config.get(
        "device",
        "cpu",
    )

    if (
        torch.cuda.is_available()
        and str(device_name) != "cpu"
    ):
        device = torch.device(
            device_name
        )
    else:
        device = torch.device(
            "cpu"
        )

    num_epochs = config.get(
        "num_epochs",
        50,
    )

    batch_size = config.get(
        "batch_size",
        16,
    )

    train_paths = config["train_paths"]
    val_paths = config["val_paths"]

    print(
        "\nStarting training trial"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Epochs: {num_epochs}"
    )

    print(
        f"Representation: "
        f"{config.get('representation', 'unknown')}"
    )

    print(
        f"Train paths: {train_paths}"
    )

    print(
        f"Validation paths: {val_paths}"
    )

    # Select dataset handling for the trainable-PCEN frontend.
    # Other frontends retain their existing chunk-based loading.
    use_trainable_pcen = (
        config.get("representation", "logmel") == "trainable_pcen"
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Load datasets
    # ----------------------------------------------------------------------------------------------------------------

    if use_trainable_pcen:
        # Paths point to the dataset root containing mel_recordings/.
        train_datasets = [
            MelRecordingDataset(path, split="train")
            for path in train_paths
        ]

        validation_datasets = [
            MelRecordingDataset(path, split="validation")
            for path in val_paths
        ]

    else:
        # Existing Log-Mel and fixed-PCEN chunk datasets.
        train_datasets = [
            load_dataset(path)
            for path in train_paths
        ]

        validation_datasets = [
            load_dataset(path)
            for path in val_paths
        ]

    train_dataset = ConcatDataset(
        train_datasets
    )

    validation_dataset = ConcatDataset(
        validation_datasets
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Normalization
    # ----------------------------------------------------------------------------------------------------------------

    feature_mean = config.get(
        "mean"
    )

    feature_std = config.get(
        "std"
    )

    # Support tensors and ordinary numeric values.
    if feature_mean is not None:
        feature_mean = torch.as_tensor(
            feature_mean,
            dtype=torch.float32,
        )

    if feature_std is not None:
        feature_std = torch.as_tensor(
            feature_std,
            dtype=torch.float32,
        )

    # Normalize explicitly in the training and validation loops.
    # TensorDataset does not automatically execute .transform.
    if (feature_mean is None) != (feature_std is None):
        raise ValueError(
            "Both mean and std must be supplied for normalization."
        )

    normalizer = (
        create_transform(
            mean=feature_mean,
            std=feature_std,
            channels_last=False,
        )
        if feature_mean is not None
        else None
    )

    # ----------------------------------------------------------------------------------------------------------------
    # DataLoaders
    # ----------------------------------------------------------------------------------------------------------------

    num_workers = config.get(
        "num_workers",
        0,
    )

    use_pin_memory = (
        torch.cuda.is_available()
        and device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1 if use_trainable_pcen else batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1 if use_trainable_pcen else batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    print(
        f"Training examples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_dataset)}"
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------------------------------------------------

    Model = config["Model"]

    model_parameters = config.get(
        "parameters",
        {},
    )

    model = Model(
        **model_parameters
    )

    model = model.to(
        device
    )

    pcen = TrainablePCEN().to(device) if use_trainable_pcen else None

    print(
        f"Model: {Model.__name__}"
    )

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------------------------------------------------------

    infrequency_weights = (
        compute_infrequency_weights(
            train_dataset
        )
    )

    infrequency_weights = (
        infrequency_weights.to(
            device
        )
    )

    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=infrequency_weights,
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Optimizer
    # ----------------------------------------------------------------------------------------------------------------

    optimizer_name = config.get(
        "optimizer",
        "AdamW",
    )

    if isinstance(
        optimizer_name,
        str,
    ):

        if optimizer_name.lower() == "adamw":

            Optimizer = optim.AdamW

        elif optimizer_name.lower() == "adam":

            Optimizer = optim.Adam

        else:

            raise ValueError(
                f"Unsupported optimizer: "
                f"{optimizer_name}"
            )

    else:

        Optimizer = optimizer_name

    learning_rate = config.get(
        "lr",
        1e-3,
    )

    weight_decay = config.get(
        "weight_decay",
        1e-4,
    )

    parameters_to_optimize = list(model.parameters())

    if use_trainable_pcen:
        parameters_to_optimize += list(pcen.parameters())

    optimizer = Optimizer(
        parameters_to_optimize,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Learning-rate scheduler
    # ----------------------------------------------------------------------------------------------------------------

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    # ----------------------------------------------------------------------------------------------------------------
    # Training state
    # ----------------------------------------------------------------------------------------------------------------

    best_micro_f1 = None

    best_macro_f1 = None

    best_class_f1 = None

    best_training_loss = None

    best_validation_loss = None

    epochs_since_improvement = 0

    # ----------------------------------------------------------------------------------------------------------------
    # Epoch loop
    # ----------------------------------------------------------------------------------------------------------------

    for epoch in range(
        num_epochs
    ):

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}"
        )

        # ============================================================================================================
        # Training
        # ============================================================================================================

        model.train()

        if use_trainable_pcen:
            pcen.train()

        total_training_loss = 0.0

        total_training_batches = 0

        for features, labels in train_loader:

            features = features.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            if use_trainable_pcen:
                # DataLoader yields [1, time, 84, 1].
                # Process the entire recording before chunking.
                if features.shape[0] != 1:
                    raise RuntimeError(
                        "Learnable PCEN requires one recording per batch."
                    )

                features = trainable_pcen_features(
                    features.squeeze(0),
                    pcen,
                )

                # Match CNN chunks: [chunks, 400, 3].
                labels = labels.reshape(-1, 400, labels.shape[-1])

                if features.shape[0] != labels.shape[0]:
                    raise RuntimeError(
                        "PCEN feature chunks and label chunks do not match."
                    )

            if normalizer is not None:
                features = normalizer(features)

            features = (
                prepare_features_for_model(
                    features
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            activations = model(
                features
            )

            loss = criterion(
                activations,
                labels,
            )

            loss = loss.mean()

            loss.backward()

            optimizer.step()

            total_training_loss += (
                loss.detach().item()
            )

            total_training_batches += 1

        training_loss = (
            total_training_loss
            / max(
                total_training_batches,
                1,
            )
        )

        # ============================================================================================================
        # Validation
        # ============================================================================================================

        model.eval()

        if use_trainable_pcen:
            pcen.eval()

        total_validation_loss = 0.0

        total_validation_batches = 0

        all_activations = []

        all_labels = []

        recording_predictions = torch.zeros(3, 3)

        with torch.no_grad():

            for features, labels in validation_loader:

                features = features.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                if use_trainable_pcen:
                    # DataLoader yields [1, time, 84, 1].
                    # Process the entire recording before chunking.
                    if features.shape[0] != 1:
                        raise RuntimeError(
                            "Learnable PCEN requires one recording per batch."
                        )

                    features = trainable_pcen_features(
                        features.squeeze(0),
                        pcen,
                    )

                    # Match CNN chunks: [chunks, 400, 3].
                    labels = labels.reshape(-1, 400, labels.shape[-1])

                    if features.shape[0] != labels.shape[0]:
                        raise RuntimeError(
                            "PCEN feature chunks and label chunks do not match."
                        )

                if normalizer is not None:
                    features = normalizer(features)

                features = (
                    prepare_features_for_model(
                        features
                    )
                )

                activations = model(
                    features
                )

                loss = criterion(
                    activations,
                    labels,
                )

                loss = loss.mean()

                total_validation_loss += (
                    loss.detach().item()
                )

                total_validation_batches += 1

                probabilities = torch.sigmoid(
                    activations.detach()
                ).cpu()

                if use_trainable_pcen:
                    recording_probabilities = probabilities.reshape(
                        1, -1, 3
                    )

                    recording_labels = labels.detach().cpu().reshape(
                        1, -1, 3
                    )

                    recording_predictions += compute_predictions(
                        compute_peaks(recording_probabilities),
                        recording_labels,
                        w=5,
                    )

                else:
                    all_activations.append(probabilities)
                    all_labels.append(labels.detach().cpu())

        validation_loss = (
            total_validation_loss
            / max(
                total_validation_batches,
                1,
            )
        )

        # ============================================================================================================
        # Event-based evaluation
        # ============================================================================================================

        if use_trainable_pcen:
            predictions = recording_predictions

        else:
            all_activations = torch.cat(
                all_activations,
                dim=0,
            )

            all_labels = torch.cat(
                all_labels,
                dim=0,
            )

            peaks = compute_peaks(
                all_activations,
                m=2,
                o=2,
                w=2,
                delta=0.1,
            )

            predictions = compute_predictions(
                peaks,
                all_labels,
                w=5,
            )

        val_f1_micro, val_f1_macro, val_f1_class = (
            f_measure(
                predictions
            )
        )

        val_f1_micro_value = float(
            val_f1_micro
        )

        val_f1_macro_value = float(
            val_f1_macro
        )

        if torch.is_tensor(
            val_f1_class
        ):

            val_f1_class_value = [
                float(value)
                for value in val_f1_class
            ]

        else:

            val_f1_class_value = [
                float(value)
                for value in val_f1_class
            ]

        # ============================================================================================================
        # Print epoch results
        # ============================================================================================================

        print(
            f"Training Loss: "
            f"{training_loss:.6f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.6f}"
        )

        print(
            f"Validation Micro F1: "
            f"{val_f1_micro_value:.6f}"
        )

        print(
            f"Validation Macro F1: "
            f"{val_f1_macro_value:.6f}"
        )

        print(
            f"Validation Class F1: "
            f"{val_f1_class_value}"
        )

        # ============================================================================================================
        # Learning-rate scheduler
        # ============================================================================================================

        scheduler.step(
            val_f1_micro_value
        )

        # ============================================================================================================
        # Best model
        # ============================================================================================================

        is_best = (
            best_micro_f1 is None
            or val_f1_micro_value > best_micro_f1
        )

        if is_best:

            best_micro_f1 = (
                val_f1_micro_value
            )

            best_macro_f1 = (
                val_f1_macro_value
            )

            best_class_f1 = (
                val_f1_class_value
            )

            best_training_loss = (
                training_loss
            )

            best_validation_loss = (
                validation_loss
            )

            epochs_since_improvement = 0

            checkpoint_path = (
                Path("/kaggle/working/temp")
                / (
                    f"adt_checkpoint_"
                    f"{os.getpid()}_"
                    f"{epoch + 1}"
                )
            )

            checkpoint_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                model.state_dict(),
                checkpoint_path / "model.pt",
            )

            if use_trainable_pcen:
                torch.save(
                    pcen.state_dict(),
                    checkpoint_path / "pcen.pt",
                )

            checkpoint = (
                train.Checkpoint.from_directory(
                    checkpoint_path.as_posix()
                )
            )

        else:

            epochs_since_improvement += 1

            checkpoint = None

        # ============================================================================================================
        # Report metrics to Ray
        # ============================================================================================================

        report_metrics = {
            "Training Loss":
                training_loss,

            "Validation Loss":
                validation_loss,

            "Micro F1":
                val_f1_micro_value,

            "Macro F1":
                val_f1_macro_value,

            "Class F1":
                val_f1_class_value,

            "best_epoch/Training Loss":
                best_training_loss,

            "best_epoch/Validation Loss":
                best_validation_loss,

            "best_epoch/Micro F1":
                best_micro_f1,

            "best_epoch/Macro F1":
                best_macro_f1,

            "best_epoch/Class F1":
                best_class_f1,

            "epochs_since_improvement":
                epochs_since_improvement,

            "epoch":
                epoch + 1,
        }

        tune.report(
            report_metrics,
            checkpoint=checkpoint,
        )

        # ============================================================================================================
        # Early stopping
        # ============================================================================================================

        early_stop = config.get(
            "early_stop",
            float("inf"),
        )

        if (
            epochs_since_improvement
            >= early_stop
        ):

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"No improvement for "
                f"{epochs_since_improvement} "
                f"epoch(s)."
            )

            break

    print(
        "\nTraining trial finished."
    )

    print(
        f"Best validation Micro F1: "
        f"{best_micro_f1}"
    )


# ----------------------------------------------------------------------------------------------------------------
# Ray Tune wrapper
# ----------------------------------------------------------------------------------------------------------------

def train_model(
    config,
    num_samples=1,
):
    """
    Run the training function through Ray Tune.

    Parameters
    ----------
    config:
        Training configuration.

    num_samples:
        Number of Ray Tune trials.
    """

    tune_config = tune.TuneConfig(
        metric="Micro F1",
        mode="max",
        num_samples=num_samples,
    )

    run_config = tune.RunConfig(
        name=(
            f"adt_"
            f"{config.get('representation', 'unknown')}"
        ),
        storage_path="/kaggle/working/ray_results",
    )

    tuner = tune.Tuner(
        tune.with_resources(
            _train_trial,
            resources={
                "cpu": 1,
                "gpu": (
                    1
                    if torch.cuda.is_available()
                    else 0
                ),
            },
        ),
        param_space=config,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

    return results

