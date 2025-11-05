"""
Training loop implementation.

Modular training loop that orchestrates batch processing, metrics,
checkpointing, and logging.
"""

from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .batch_processor import BatchProcessor
from .checkpoint_manager import CheckpointManager
from .metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""

    epoch: int
    train_loss: float
    train_metrics: Dict[str, float]
    val_loss: Optional[float] = None
    val_metrics: Optional[Dict[str, float]] = None


class TrainingLoop:
    """
    Modular training loop for interactive segmentation.

    Orchestrates:
    - Training/validation epochs
    - Batch processing
    - Metrics tracking
    - Checkpointing
    - Logging

    This is a cleaner, more modular alternative to the monolithic ISTrainer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_processor: BatchProcessor,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        device: Optional[torch.device] = None,
        gradient_clip_norm: Optional[float] = None,
        log_every_n_batches: int = 10,
    ):
        """
        Initialize training loop.

        Args:
            model: Model to train
            optimizer: Optimizer
            batch_processor: Handles batch processing
            train_loader: Training data loader
            val_loader: Validation data loader
            scheduler: Learning rate scheduler
            checkpoint_manager: Checkpoint manager
            device: Device to train on
            gradient_clip_norm: Gradient clipping norm
            log_every_n_batches: Log frequency
        """
        self.model = model
        self.optimizer = optimizer
        self.batch_processor = batch_processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gradient_clip_norm = gradient_clip_norm
        self.log_every_n_batches = log_every_n_batches

        # Metrics tracking
        self.train_metrics_tracker = MetricsTracker()
        self.val_metrics_tracker = MetricsTracker()

        # Move model to device
        self.model.to(self.device)

        self.current_epoch = 0

    def run(
        self,
        num_epochs: int,
        start_epoch: int = 0,
        callbacks: Optional[Dict[str, Callable]] = None,
    ) -> List[EpochMetrics]:
        """
        Run training for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch number
            callbacks: Optional callbacks dict with keys:
                - 'on_epoch_start'
                - 'on_epoch_end'
                - 'on_batch_end'

        Returns:
            List of epoch metrics
        """
        callbacks = callbacks or {}
        epoch_metrics_list = []

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training batches: {len(self.train_loader)}")
        if self.val_loader:
            logger.info(f"Validation batches: {len(self.val_loader)}")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Callback: epoch start
            if "on_epoch_start" in callbacks:
                callbacks["on_epoch_start"](epoch)

            # Training epoch
            train_loss, train_metrics = self._train_epoch(epoch, callbacks)

            # Validation epoch
            val_loss = None
            val_metrics = None
            if self.val_loader is not None:
                val_loss, val_metrics = self._validate_epoch(epoch)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    # Needs metric
                    metric_val = val_loss if val_loss is not None else train_loss
                    self.scheduler.step(metric_val)
                else:
                    self.scheduler.step()

            # Create epoch metrics
            epoch_result = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_metrics=train_metrics,
                val_loss=val_loss,
                val_metrics=val_metrics,
            )
            epoch_metrics_list.append(epoch_result)

            # Log epoch summary
            self._log_epoch_summary(epoch_result)

            # Checkpointing
            if self.checkpoint_manager is not None:
                self._handle_checkpointing(epoch, epoch_result)

            # Callback: epoch end
            if "on_epoch_end" in callbacks:
                callbacks["on_epoch_end"](epoch, epoch_result)

        logger.info("Training completed!")
        return epoch_metrics_list

    def _train_epoch(self, epoch: int, callbacks: Dict[str, Callable]) -> tuple:
        """Run one training epoch."""
        self.model.train()
        self.batch_processor.reset_metrics()
        self.train_metrics_tracker = MetricsTracker()

        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", total=num_batches)

        for batch_idx, batch in enumerate(pbar):
            # Process batch
            loss, loss_dict, batch_data, outputs = self.batch_processor.process_batch(
                batch, device=self.device, is_training=True
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            self.train_metrics_tracker.update(loss_dict)

            # Update progress bar
            if (batch_idx + 1) % self.log_every_n_batches == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Callback: batch end
            if "on_batch_end" in callbacks:
                callbacks["on_batch_end"](epoch, batch_idx, loss_dict)

        # Compute epoch averages
        avg_loss = total_loss / num_batches
        train_metrics = self.train_metrics_tracker.end_epoch()

        return avg_loss, train_metrics

    def _validate_epoch(self, epoch: int) -> tuple:
        """Run one validation epoch."""
        self.model.eval()
        self.batch_processor.reset_metrics()
        self.val_metrics_tracker = MetricsTracker()

        total_loss = 0.0
        num_batches = len(self.val_loader)

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", total=num_batches)

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Process batch
                loss, loss_dict, batch_data, outputs = (
                    self.batch_processor.process_batch(
                        batch, device=self.device, is_training=False
                    )
                )

                # Track metrics
                total_loss += loss.item()
                self.val_metrics_tracker.update(loss_dict)

                # Update progress bar
                if (batch_idx + 1) % self.log_every_n_batches == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Compute epoch averages
        avg_loss = total_loss / num_batches
        val_metrics = self.val_metrics_tracker.end_epoch()

        # Get metrics from batch processor
        batch_metrics = self.batch_processor.get_metrics()
        val_metrics.update(batch_metrics)

        return avg_loss, val_metrics

    def _log_epoch_summary(self, epoch_result: EpochMetrics):
        """Log summary of epoch."""
        log_msg = f"\n{'=' * 80}\n"
        log_msg += f"Epoch {epoch_result.epoch} Summary:\n"
        log_msg += f"  Train Loss: {epoch_result.train_loss:.4f}\n"

        if epoch_result.train_metrics:
            log_msg += "  Train Metrics:\n"
            for name, value in epoch_result.train_metrics.items():
                log_msg += f"    {name}: {value:.4f}\n"

        if epoch_result.val_loss is not None:
            log_msg += f"  Val Loss: {epoch_result.val_loss:.4f}\n"

        if epoch_result.val_metrics:
            log_msg += "  Val Metrics:\n"
            for name, value in epoch_result.val_metrics.items():
                log_msg += f"    {name}: {value:.4f}\n"

        log_msg += f"{'=' * 80}\n"
        logger.info(log_msg)

    def _handle_checkpointing(self, epoch: int, metrics: EpochMetrics):
        """Handle checkpoint saving."""
        # Periodic checkpoint
        if self.checkpoint_manager.should_save_checkpoint(epoch):
            self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics={
                    "train_loss": metrics.train_loss,
                    "val_loss": metrics.val_loss,
                },
            )

        # Best checkpoint (based on validation loss if available)
        if metrics.val_loss is not None:
            self.checkpoint_manager.update_best_checkpoint(
                metric_value=metrics.val_loss,
                epoch=epoch,
                model=self.model,
                higher_is_better=False,  # Lower loss is better
            )

    def resume_from_checkpoint(self, checkpoint_path: Path) -> int:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Epoch to resume from
        """
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Resuming from epoch {start_epoch}")

        return start_epoch
