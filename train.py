import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset

from ray import train, tune
from ray.train import Checkpoint

from preprocess import compute_infrequency_weights, create_transform
from evaluate import compute_peaks, compute_predictions, f_measure

from tempfile import TemporaryDirectory
from pathlib import Path
from copy import deepcopy


def train_model(config: tune.TuneConfig):
    """ Training function to use with RayTune """
    # Seed torch and cuda
    seed = config["seed"]
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # Declare device
    device = config["device"] if torch.cuda.is_available() else "cpu"
    print(f"Training: Can use CUDA: {torch.cuda.is_available()}")

    # Load the datasets into dataloaders
    train_loader = DataLoader(ConcatDataset(map(torch.load, config["train_paths"])), shuffle=True, batch_size=config["batch_size"], num_workers=4, pin_memory=True)
    val_loader = DataLoader(ConcatDataset(map(torch.load, config["val_paths"])), shuffle=True, batch_size=config["batch_size"], num_workers=4, pin_memory=True)

    # Create a transform preprocessing pipeline
    transforms = create_transform(**config["transforms"], channels_last=True)

    # Create the model, loss function and optimizer
    model = config["Model"](**config["parameters"]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = config["optimizer"](model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    optimizer.zero_grad(set_to_none=True)
    print("Number of parameters: ", sum(param.numel() for param in model.parameters()))

    # Add a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)

    # Compute infrequent instrument weights from the training dataset
    infrequency_weights = compute_infrequency_weights(train_loader).to(device)
    print("Infrequency weights: ", infrequency_weights)
    
    # Start training
    print(f"Started training on {device}")
    epochs_since_improvement, best_epoch, val_loss_best, val_f1_micro_best = 0, None, None, None
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss, n_batches_train = 0.0, 0
        for i, data in enumerate(train_loader):
            # Perform forward, backward and optimization step
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(transforms(inputs))

            # Compute per timestep weights given labels and infrequency weights
            timestep_weights = (labels * infrequency_weights).sum(dim=2)
            timestep_weights = torch.max(torch.tensor(1.0), timestep_weights)

            loss = (loss_fn(outputs, labels).sum(dim=2) * timestep_weights).mean()
            loss.backward()

            # Clip the gradients to prevent explosions
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, error_if_nonfinite=True)

            optimizer.step()
            optimizer.zero_grad()

            # And store training loss
            train_loss += loss.item()
            n_batches_train += 1


        # After a training epoch, compute validation performance
        model.eval()
        val_loss, n_batches_val = 0.0, 0
        val_predictions = torch.zeros(size=(infrequency_weights.shape[0], 3))
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(transforms(inputs))

                loss = loss_fn(outputs, labels).sum(dim=2).mean()
                val_loss += loss.item()
                n_batches_val += 1

                # Compute activation probabilties and add to predictions
                activations = F.sigmoid(outputs)
                val_predictions += compute_predictions(compute_peaks(activations), labels)

                if i == 0:
                    print(torch.stack((F.sigmoid(outputs), labels), dim=-1))
                    
        # Average the losses
        train_loss /= n_batches_train
        val_loss /= n_batches_val

        # Step the scheduler
        scheduler.step(val_loss)
        print("Learning rate:", scheduler.get_last_lr())

        # Compute F1 score
        val_f1_micro, val_f1_macro, val_f1_class = f_measure(val_predictions)
        print("Predictions:", val_predictions)
        print("Class F1s:", val_f1_class)

        # Check if we should increment early stop count
        if val_loss_best is None or val_loss < val_loss_best:
            val_loss_best = val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            
        # Check if we should checkpoint current model by comparing micro F1 score
        with TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            checkpoint_path = Path(temp_checkpoint_dir)
            if val_f1_micro_best is None or val_f1_micro > val_f1_micro_best:
                val_f1_micro_best = val_f1_micro

                # Save model to the temporary checkpoint directory
                model_path = checkpoint_path / "model.pt"
                torch.save(model.state_dict(), model_path)
                
                # Create a Checkpoint object
                checkpoint = Checkpoint.from_directory(checkpoint_path)

                # Store best epoch information
                best_epoch = {
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss,
                    "Micro F1": val_f1_micro.item(),
                    "Macro F1": val_f1_macro.item(),
                    "Class F1": val_f1_class.tolist(),
                }

            # Report to RayTune
            train.report({
                "Training Loss": train_loss,
                "Validation Loss": val_loss,
                "Micro F1": val_f1_micro.item(),
                "Macro F1": val_f1_macro.item(),
                "Class F1": val_f1_class.tolist(),
                "best_epoch": best_epoch,
                "epochs_since_improvement": epochs_since_improvement
                },
                checkpoint=checkpoint
            )
    print("Finished training")
