"""
Training callbacks for segmentation experiments.

Provides reusable callback classes for early stopping, model checkpointing,
and training metric logging.

Usage:
    from _common.callbacks import EarlyStopping, ModelCheckpoint, TrainingLogger

    early_stop = EarlyStopping(patience=15, monitor="val_dice", mode="max")
    checkpoint = ModelCheckpoint(save_dir="checkpoints/", monitor="val_dice", mode="max")
    logger = TrainingLogger(log_dir="runs/")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(...)
        val_metrics = validate(...)

        logger.log_epoch(epoch, train_loss, val_metrics)
        checkpoint.step(epoch, val_metrics["val_dice"], model, optimizer)

        if early_stop.step(val_metrics["val_dice"]):
            print(f"Early stopping at epoch {epoch}")
            break
"""

import json
import csv
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        monitor: Name of the metric to monitor (for logging purposes).
        mode: One of 'min' or 'max'. In 'min' mode, training stops when the
              metric has stopped decreasing; in 'max' mode, when it has stopped
              increasing.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value: Optional[float] = None
        self.should_stop = False

        if mode == "min":
            self._is_better = lambda current, best: current < best - min_delta
        elif mode == "max":
            self._is_better = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    def step(self, metric_value: float) -> bool:
        """Check whether training should stop.

        Args:
            metric_value: Current value of the monitored metric.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_value is None or self._is_better(metric_value, self.best_value):
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience
        return self.should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class ModelCheckpoint:
    """Save model checkpoints based on a monitored metric.

    Saves the best model (based on the monitored metric) and optionally
    saves periodic checkpoints.

    Args:
        save_dir: Directory to save checkpoints.
        monitor: Name of the metric to monitor.
        mode: One of 'min' or 'max'.
        save_best_only: If True, only save when the metric improves.
        save_interval: Save a checkpoint every N epochs (0 to disable).
        filename_prefix: Prefix for checkpoint filenames.
    """

    def __init__(
        self,
        save_dir: Union[str, Path] = "checkpoints",
        monitor: str = "val_dice",
        mode: str = "max",
        save_best_only: bool = True,
        save_interval: int = 0,
        filename_prefix: str = "model",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_interval = save_interval
        self.filename_prefix = filename_prefix
        self.best_value: Optional[float] = None

        if mode == "min":
            self._is_better = lambda current, best: current < best
        elif mode == "max":
            self._is_better = lambda current, best: current > best
        else:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

    def step(
        self,
        epoch: int,
        metric_value: float,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Evaluate whether to save a checkpoint.

        Args:
            epoch: Current epoch number.
            metric_value: Current value of the monitored metric.
            model: The model to save.
            optimizer: Optional optimizer to include in the checkpoint.
            extra: Optional extra data to include in the checkpoint.

        Returns:
            True if a checkpoint was saved, False otherwise.
        """
        saved = False

        # Check if this is the best so far
        is_best = self.best_value is None or self._is_better(metric_value, self.best_value)
        if is_best:
            self.best_value = metric_value

        # Save best model
        if is_best:
            self._save(epoch, metric_value, model, optimizer, extra, tag="best")
            saved = True

        # Save periodic checkpoint
        if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
            if not (self.save_best_only and not is_best):
                self._save(epoch, metric_value, model, optimizer, extra, tag=f"epoch{epoch:04d}")
                saved = True

        return saved

    def _save(
        self,
        epoch: int,
        metric_value: float,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        extra: Optional[Dict[str, Any]],
        tag: str,
    ):
        """Save a checkpoint to disk."""
        checkpoint = {
            "epoch": epoch,
            self.monitor: metric_value,
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if extra is not None:
            checkpoint.update(extra)

        path = self.save_dir / f"{self.filename_prefix}_{tag}.pth"
        torch.save(checkpoint, path)


class TrainingLogger:
    """Log training metrics to CSV and optionally to TensorBoard.

    Args:
        log_dir: Directory for log files.
        csv_filename: Name of the CSV log file.
        use_tensorboard: Whether to log to TensorBoard as well.
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "runs",
        csv_filename: str = "training_log.csv",
        use_tensorboard: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / csv_filename
        self._csv_initialized = False

        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("Warning: TensorBoard not available. Logging to CSV only.")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ):
        """Log metrics for one epoch.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for this epoch.
            val_metrics: Optional dictionary of validation metrics.
            lr: Optional current learning rate.
        """
        row = {"epoch": epoch, "train_loss": train_loss}
        if lr is not None:
            row["lr"] = lr
        if val_metrics:
            row.update(val_metrics)

        # Write to CSV
        self._write_csv_row(row)

        # Write to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            if lr is not None:
                self.writer.add_scalar("LR", lr, epoch)
            if val_metrics:
                for name, value in val_metrics.items():
                    self.writer.add_scalar(f"Metrics/{name}", value, epoch)

    def _write_csv_row(self, row: Dict[str, Any]):
        """Append a row to the CSV log file."""
        fieldnames = list(row.keys())

        if not self._csv_initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self._csv_initialized = True
        else:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
