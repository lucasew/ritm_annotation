"""
Metrics tracking for training.

Handles accumulation and logging of training/validation metrics.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks and aggregates metrics during training.

    Features:
    - Accumulate metrics over batches
    - Compute epoch averages
    - Track best values
    - Simple history tracking
    """

    def __init__(self, metric_names: Optional[List[str]] = None):
        """
        Initialize metrics tracker.

        Args:
            metric_names: List of metric names to track
        """
        self.metric_names = metric_names or []

        # Current epoch accumulation
        self._current_epoch: Dict[str, List[float]] = defaultdict(list)

        # History: epoch -> metric -> value
        self._history: List[Dict[str, float]] = []

        # Best values: metric -> (value, epoch)
        self._best_values: Dict[str, tuple] = {}

    def update(self, metrics: Dict[str, float]):
        """
        Update with new metric values.

        Args:
            metrics: Dictionary of metric_name -> value
        """
        for name, value in metrics.items():
            self._current_epoch[name].append(value)

    def end_epoch(self) -> Dict[str, float]:
        """
        End current epoch and compute averages.

        Returns:
            Dictionary of averaged metrics
        """
        epoch_metrics = {}

        for name, values in self._current_epoch.items():
            if values:
                avg_value = sum(values) / len(values)
                epoch_metrics[name] = avg_value

                # Update best values
                if name not in self._best_values:
                    self._best_values[name] = (avg_value, len(self._history))
                else:
                    best_val, best_epoch = self._best_values[name]
                    # Assume higher is better for now
                    # TODO: Make this configurable
                    if avg_value > best_val:
                        self._best_values[name] = (avg_value, len(self._history))

        # Store in history
        self._history.append(epoch_metrics)

        # Reset current epoch
        self._current_epoch.clear()

        return epoch_metrics

    def get_current_avg(self) -> Dict[str, float]:
        """Get current running average without ending epoch."""
        avg_metrics = {}
        for name, values in self._current_epoch.items():
            if values:
                avg_metrics[name] = sum(values) / len(values)
        return avg_metrics

    def get_best_value(self, metric_name: str) -> Optional[tuple]:
        """
        Get best value for a metric.

        Args:
            metric_name: Name of metric

        Returns:
            Tuple of (value, epoch) or None
        """
        return self._best_values.get(metric_name)

    def get_history(self) -> List[Dict[str, float]]:
        """Get full history of epoch metrics."""
        return self._history

    def get_last_epoch_metrics(self) -> Optional[Dict[str, float]]:
        """Get metrics from last completed epoch."""
        if self._history:
            return self._history[-1]
        return None

    def reset(self):
        """Reset all tracking."""
        self._current_epoch.clear()
        self._history.clear()
        self._best_values.clear()

    def log_epoch_metrics(self, epoch: int, prefix: str = ""):
        """
        Log epoch metrics.

        Args:
            epoch: Epoch number
            prefix: Prefix for log messages (e.g., "train", "val")
        """
        metrics = self.get_last_epoch_metrics()
        if metrics:
            metrics_str = ", ".join([
                f"{name}: {value:.4f}"
                for name, value in metrics.items()
            ])
            logger.info(f"Epoch {epoch} {prefix} metrics: {metrics_str}")
