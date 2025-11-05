"""
Checkpoint management for training.

Handles saving and loading model checkpoints with metadata.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Features:
    - Save/load model state
    - Save/load optimizer state
    - Save/load scheduler state
    - Keep best checkpoints based on metrics
    - Periodic checkpoint saving
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        model_name: str = "model",
        keep_best_n: int = 3,
        save_every_n_epochs: int = 10,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            model_name: Name prefix for checkpoint files
            keep_best_n: Number of best checkpoints to keep
            save_every_n_epochs: Save checkpoint every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.keep_best_n = keep_best_n
        self.save_every_n_epochs = save_every_n_epochs

        self.best_checkpoints = []  # List of (metric_value, path)

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer to save
            scheduler: LR scheduler to save
            metrics: Current metrics
            extra_data: Any extra data to save
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if metrics is not None:
            checkpoint['metrics'] = metrics

        if extra_data is not None:
            checkpoint.update(extra_data)

        # Determine checkpoint path
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_{epoch:03d}.pth"

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Always save latest
        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pth"
        torch.save(checkpoint, latest_path)

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into

        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model state from {checkpoint_path}")

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded optimizer state")

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"Loaded scheduler state")

        return checkpoint

    def should_save_checkpoint(self, epoch: int) -> bool:
        """Check if checkpoint should be saved this epoch."""
        return (epoch + 1) % self.save_every_n_epochs == 0

    def update_best_checkpoint(
        self,
        metric_value: float,
        epoch: int,
        model: torch.nn.Module,
        higher_is_better: bool = True,
    ):
        """
        Update best checkpoints based on metric.

        Args:
            metric_value: Current metric value
            epoch: Current epoch
            model: Model to potentially save
            higher_is_better: Whether higher metric is better
        """
        # Check if this is a best checkpoint
        is_best = False

        if len(self.best_checkpoints) < self.keep_best_n:
            is_best = True
        else:
            # Check against worst of the best
            worst_best = min(self.best_checkpoints, key=lambda x: x[0])
            if higher_is_better:
                is_best = metric_value > worst_best[0]
            else:
                is_best = metric_value < worst_best[0]

        if is_best:
            # Save checkpoint
            checkpoint_path = self.save_checkpoint(
                epoch=epoch,
                model=model,
                is_best=False,
            )

            # Update best list
            self.best_checkpoints.append((metric_value, checkpoint_path))
            self.best_checkpoints.sort(
                key=lambda x: x[0],
                reverse=higher_is_better
            )

            # Remove worst if over limit
            if len(self.best_checkpoints) > self.keep_best_n:
                _, path_to_remove = self.best_checkpoints.pop()
                if path_to_remove.exists():
                    path_to_remove.unlink()

            # Save as best
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            checkpoint_path.rename(best_path)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint if it exists."""
        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pth"
        return latest_path if latest_path.exists() else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint if it exists."""
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        return best_path if best_path.exists() else None
