"""
Batch processing logic for interactive segmentation training.

Handles the forward pass, loss computation, and click simulation.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np

from ...model.is_model import ISModel


class BatchProcessor:
    """
    Processes batches for interactive segmentation training.

    Handles:
    - Iterative click simulation
    - Forward passes with progressive refinement
    - Loss computation
    - Metric updates
    """

    def __init__(
        self,
        model: ISModel,
        loss_fn,
        metrics: Optional[list] = None,
        max_interactive_points: int = 0,
        click_models: Optional[list] = None,
    ):
        """
        Initialize batch processor.

        Args:
            model: Interactive segmentation model
            loss_fn: Loss function(s) configuration
            metrics: List of metric objects
            max_interactive_points: Max number of simulated clicks
            click_models: Models for simulating clicks
        """
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.max_interactive_points = max_interactive_points
        self.click_models = click_models or []

    def process_batch(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
        """
        Process a single batch.

        Args:
            batch: Dictionary with 'images', 'points', 'instances'
            device: Device to run on
            is_training: Whether in training mode

        Returns:
            Tuple of (total_loss, loss_dict, batch_data, outputs)
        """
        # Move batch to device
        batch = self._move_to_device(batch, device)

        # Initialize outputs
        outputs = {}
        total_loss = 0.0
        losses_logging = {}

        # Simulate interactive refinement
        num_clicks = self._get_num_clicks(is_training)

        prev_output = None
        for click_idx in range(num_clicks + 1):
            # Get current points
            if click_idx == 0:
                points = batch["points"]
            else:
                # Simulate next click based on error
                points = self._simulate_next_clicks(batch, prev_output, click_idx)

            # Forward pass
            net_input = self._prepare_input(batch, points, prev_output)
            output = self.model(**net_input)

            # Store outputs
            if click_idx == num_clicks:
                outputs["instances"] = output

            prev_output = output.detach()

        # Compute losses
        loss, losses_info = self._compute_losses(outputs, batch, is_training)

        total_loss = loss
        losses_logging.update(losses_info)

        # Update metrics
        if not is_training and self.metrics:
            self._update_metrics(outputs, batch)

        return total_loss, losses_logging, batch, outputs

    def _move_to_device(
        self, batch: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _get_num_clicks(self, is_training: bool) -> int:
        """Get number of clicks to simulate."""
        if is_training and self.max_interactive_points > 0:
            return np.random.randint(0, self.max_interactive_points + 1)
        return 0

    def _prepare_input(
        self,
        batch: Dict[str, torch.Tensor],
        points: torch.Tensor,
        prev_output: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare model input."""
        net_input = {
            "image": batch["images"],
            "points": points,
        }

        if prev_output is not None and self.model.with_prev_mask:
            net_input["prev_mask"] = prev_output

        return net_input

    def _simulate_next_clicks(
        self, batch: Dict[str, torch.Tensor], prev_output: torch.Tensor, click_idx: int
    ) -> torch.Tensor:
        """
        Simulate next clicks based on current predictions.

        Uses distance transform to place clicks at hard locations.
        """
        from ...model.is_model import get_next_points

        points = batch["points"]

        # Get next points using error-based sampling
        next_points = get_next_points(
            prev_output, batch["instances"], points, click_idx + 1
        )

        return next_points

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        is_training: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute losses from outputs.

        Args:
            outputs: Model outputs
            batch: Batch data with ground truth
            is_training: Whether in training mode

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses_logging = {}
        total_loss = 0.0

        # Instance segmentation loss
        if "instances" in outputs:
            instance_loss = self.loss_fn["instance_loss"](
                outputs["instances"], batch["instances"]
            )
            total_loss += instance_loss
            losses_logging["instance_loss"] = instance_loss.item()

            # Auxiliary loss (if model has aux output)
            if hasattr(self.model, "with_aux_output") and self.model.with_aux_output:
                if "instances_aux" in outputs:
                    aux_loss = self.loss_fn["instance_aux_loss"](
                        outputs["instances_aux"], batch["instances"]
                    )
                    total_loss += 0.4 * aux_loss
                    losses_logging["instance_aux_loss"] = aux_loss.item()

        return total_loss, losses_logging

    def _update_metrics(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ):
        """Update metrics with predictions."""
        for metric in self.metrics:
            metric.update(outputs["instances"], batch["instances"])

    def reset_metrics(self):
        """Reset all metrics for new epoch."""
        for metric in self.metrics:
            if hasattr(metric, "reset_epoch_stats"):
                metric.reset_epoch_stats()

    def get_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        metrics_dict = {}
        for metric in self.metrics:
            if hasattr(metric, "get_epoch_value"):
                value = metric.get_epoch_value()
                name = getattr(metric, "name", metric.__class__.__name__)
                metrics_dict[name] = value
        return metrics_dict
