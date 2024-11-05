from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.base import TorchModel


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(
    model: TorchModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion=None,
    optimizer=None,
    num_epochs: int = 20,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_path: Path | None = None,
    early_stopping_patience: int = 5,
    use_scheduler: bool = True,
) -> nn.Module:
    """
    Unified training function that combines features from both implementations.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function (if None, uses model.criterion)
        optimizer: Optimizer (if None, uses model.optimizer)
        num_epochs: Number of epochs to train
        device: Device to train on
        save_path: Path to save best model checkpoint
        early_stopping_patience: Number of epochs to wait before early stopping
        use_scheduler: Whether to use OneCycleLR scheduler

    Returns:
        Trained model
    """
    # Setup device
    model = model.to(device)

    # Handle criterion and optimizer
    criterion = criterion if criterion is not None else getattr(model, "criterion", None)
    optimizer = optimizer if optimizer is not None else getattr(model, "optimizer", None)

    if criterion is None or optimizer is None:
        raise ValueError(
            "Either provide criterion and optimizer or ensure model has them as attributes"
        )

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    # Initialize scheduler if requested
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0,
        )

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Handle both dictionary and tensor inputs
            if isinstance(images, dict):
                images = {k: v.to(device) for k, v in images.items()}
                outputs = model(**images).logits
            else:
                images = images.to(device)
                outputs = model(images)

            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            if scheduler:
                scheduler.step()

            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{train_loss/(batch_idx+1):.4f}",
                    "acc": f"{train_correct/train_total:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                }
            )

        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                if isinstance(images, dict):
                    images = {k: v.to(device) for k, v in images.items()}
                    outputs = model(**images).logits
                else:
                    images = images.to(device)
                    outputs = model(images)

                labels = labels.to(device)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Print epoch results
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Save best model
        if val_acc > best_val_acc and save_path:
            best_val_acc = val_acc
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "loss": val_loss,
                    "accuracy": val_acc,
                },
                save_path,
            )

    return model


def validate_model(
    model: TorchModel,
    val_loader: DataLoader,
    device: str,
) -> dict:
    """Validate model performance"""
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for batch, labels in val_loader:
            labels = labels.to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model.model(**batch).logits
            loss = model.criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return {
        "accuracy": correct / total if total > 0 else 0,
        "total_samples": total,
        "loss": val_loss / len(val_loader) if len(val_loader) > 0 else float("inf"),
    }
