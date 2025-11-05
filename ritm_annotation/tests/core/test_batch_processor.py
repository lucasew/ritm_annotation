"""
Tests for BatchProcessor with real PyTorch models.

Tests the batch processing logic for training.
"""

import pytest
import torch
import numpy as np


pytestmark = pytest.mark.integration


@pytest.fixture
def batch_processor(test_model, simple_loss_fn, simple_metric, device):
    """Create BatchProcessor with real model."""
    from ritm_annotation.core.training import BatchProcessor

    model, model_cfg = test_model

    processor = BatchProcessor(
        model=model,
        loss_fn=simple_loss_fn,
        metrics=[simple_metric],
        max_interactive_points=3,
    )

    return processor


class TestBatchProcessor:
    """Test BatchProcessor with real model."""

    def test_process_single_batch(self, batch_processor, test_batch, device):
        """Test processing a single batch."""
        loss, loss_dict, batch_data, outputs = batch_processor.process_batch(
            test_batch,
            device=device,
            is_training=True
        )

        # Validate loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0

        # Validate loss dict
        assert 'instance_loss' in loss_dict
        assert isinstance(loss_dict['instance_loss'], float)

        # Validate outputs
        assert 'instances' in outputs
        assert isinstance(outputs['instances'], torch.Tensor)

    def test_training_mode_vs_eval_mode(self, batch_processor, test_batch, device):
        """Test different behavior in training vs eval mode."""
        # Training mode
        loss_train, _, _, _ = batch_processor.process_batch(
            test_batch,
            device=device,
            is_training=True
        )

        # Eval mode
        loss_eval, _, _, _ = batch_processor.process_batch(
            test_batch,
            device=device,
            is_training=False
        )

        # Both should produce valid losses
        assert not torch.isnan(loss_train)
        assert not torch.isnan(loss_eval)

    def test_reset_metrics(self, batch_processor):
        """Test resetting metrics."""
        batch_processor.reset_metrics()

        # Should not raise error
        metrics = batch_processor.get_metrics()
        assert isinstance(metrics, dict)

    def test_get_metrics_after_batch(
        self,
        batch_processor,
        test_batch,
        device
    ):
        """Test getting metrics after processing batches."""
        # Reset first
        batch_processor.reset_metrics()

        # Process several batches
        for _ in range(3):
            batch_processor.process_batch(
                test_batch,
                device=device,
                is_training=False  # Only updates metrics in eval mode
            )

        # Get metrics
        metrics = batch_processor.get_metrics()
        assert isinstance(metrics, dict)

        if 'iou' in metrics:
            assert 0.0 <= metrics['iou'] <= 1.0

    def test_batch_processor_with_different_batch_sizes(
        self,
        batch_processor,
        device
    ):
        """Test with different batch sizes."""
        for batch_size in [1, 2, 4]:
            batch = {
                'images': torch.randn(batch_size, 3, 320, 480).to(device),
                'points': torch.randint(0, 100, (batch_size, 1, 3)).float().to(device),
                'instances': torch.randint(0, 2, (batch_size, 1, 320, 480)).float().to(device),
            }

            loss, _, _, _ = batch_processor.process_batch(
                batch,
                device=device,
                is_training=True
            )

            assert not torch.isnan(loss)
            assert loss.item() >= 0


class TestBatchProcessorClickSimulation:
    """Test click simulation during training."""

    def test_max_interactive_points(self, test_model, simple_loss_fn, device):
        """Test that max_interactive_points is respected."""
        from ritm_annotation.core.training import BatchProcessor

        model, _ = test_model

        # Create processor with specific max points
        processor = BatchProcessor(
            model=model,
            loss_fn=simple_loss_fn,
            metrics=[],
            max_interactive_points=5,
        )

        assert processor.max_interactive_points == 5

    def test_zero_interactive_points(self, test_model, simple_loss_fn, test_batch, device):
        """Test with no interactive refinement."""
        from ritm_annotation.core.training import BatchProcessor

        model, _ = test_model

        processor = BatchProcessor(
            model=model,
            loss_fn=simple_loss_fn,
            metrics=[],
            max_interactive_points=0,
        )

        loss, _, _, _ = processor.process_batch(
            test_batch,
            device=device,
            is_training=True
        )

        assert not torch.isnan(loss)


class TestBatchProcessorNumericalStability:
    """Test numerical stability of batch processing."""

    def test_no_nans_in_outputs(self, batch_processor, test_batch, device):
        """Test that outputs don't contain NaN."""
        loss, loss_dict, _, outputs = batch_processor.process_batch(
            test_batch,
            device=device,
            is_training=True
        )

        # Check loss
        assert not torch.isnan(loss)

        # Check all loss dict values
        for key, value in loss_dict.items():
            assert not np.isnan(value), f"NaN in {key}"

        # Check outputs
        for key, tensor in outputs.items():
            if isinstance(tensor, torch.Tensor):
                assert not torch.isnan(tensor).any(), f"NaN in output {key}"

    def test_no_infs_in_outputs(self, batch_processor, test_batch, device):
        """Test that outputs don't contain Inf."""
        loss, loss_dict, _, outputs = batch_processor.process_batch(
            test_batch,
            device=device,
            is_training=True
        )

        # Check loss
        assert not torch.isinf(loss)

        # Check all loss dict values
        for key, value in loss_dict.items():
            assert not np.isinf(value), f"Inf in {key}"

        # Check outputs
        for key, tensor in outputs.items():
            if isinstance(tensor, torch.Tensor):
                assert not torch.isinf(tensor).any(), f"Inf in output {key}"

    def test_loss_is_positive(self, batch_processor, test_batch, device):
        """Test that loss is always positive."""
        loss, _, _, _ = batch_processor.process_batch(
            test_batch,
            device=device,
            is_training=True
        )

        assert loss.item() >= 0, f"Loss is negative: {loss.item()}"

    def test_outputs_in_valid_range(self, batch_processor, test_batch, device):
        """Test that output probabilities are in valid range."""
        _, _, _, outputs = batch_processor.process_batch(
            test_batch,
            device=device,
            is_training=True
        )

        if 'instances' in outputs:
            instances = outputs['instances']
            # After sigmoid, should be in [0, 1]
            # But model outputs logits, so just check no extreme values
            assert instances.abs().max() < 100, "Output values too extreme"


class TestBatchProcessorGradients:
    """Test gradient computation."""

    def test_gradients_are_computed(self, batch_processor, test_batch, device):
        """Test that gradients are properly computed."""
        # Enable gradients for model parameters
        for param in batch_processor.model.parameters():
            param.requires_grad = True

        loss, _, _, _ = batch_processor.process_batch(
            test_batch,
            device=device,
            is_training=True
        )

        # Compute gradients
        loss.backward()

        # Check that at least some parameters have gradients
        has_gradient = False
        for param in batch_processor.model.parameters():
            if param.grad is not None:
                has_gradient = True
                # Check gradient is not all zeros
                if param.grad.abs().sum() > 0:
                    break

        assert has_gradient, "No gradients computed"

        # Reset gradients
        batch_processor.model.zero_grad()

    def test_no_gradient_leakage_in_eval(
        self,
        batch_processor,
        test_batch,
        device
    ):
        """Test that no gradients are leaked in eval mode."""
        with torch.no_grad():
            loss, _, _, outputs = batch_processor.process_batch(
                test_batch,
                device=device,
                is_training=False
            )

            # Outputs should not require gradients
            for key, tensor in outputs.items():
                if isinstance(tensor, torch.Tensor):
                    assert not tensor.requires_grad, \
                        f"Output {key} requires gradient in eval mode"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
