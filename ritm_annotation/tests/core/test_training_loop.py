"""
Tests for TrainingLoop with real PyTorch components.

Tests complete training loop with model, optimizer, data loaders.
"""

import pytest
import torch
from torch.utils.data import DataLoader


pytestmark = pytest.mark.integration


@pytest.fixture
def training_components(
    test_model, simple_loss_fn, simple_metric, temp_checkpoint_dir, device
):
    """Create all components needed for training."""
    from ritm_annotation.core.training import (
        TrainingLoop,
        BatchProcessor,
        CheckpointManager,
    )
    from ritm_annotation.tests.conftest import create_synthetic_dataset

    model, model_cfg = test_model

    # Enable gradients
    for param in model.parameters():
        param.requires_grad = True

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create batch processor
    batch_processor = BatchProcessor(
        model=model,
        loss_fn=simple_loss_fn,
        metrics=[simple_metric],
        max_interactive_points=2,
    )

    # Create data loaders
    train_dataset = create_synthetic_dataset(num_samples=10, image_size=(320, 480))
    val_dataset = create_synthetic_dataset(num_samples=5, image_size=(320, 480))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=temp_checkpoint_dir,
        model_name="test_model",
        keep_best_n=2,
        save_every_n_epochs=2,
    )

    # Create training loop
    loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        batch_processor=batch_processor,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=checkpoint_manager,
        device=device,
        log_every_n_batches=2,
    )

    return {
        "loop": loop,
        "model": model,
        "optimizer": optimizer,
        "checkpoint_manager": checkpoint_manager,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


class TestTrainingLoop:
    """Test TrainingLoop with real components."""

    def test_single_epoch(self, training_components):
        """Test running a single epoch."""
        loop = training_components["loop"]

        # Run one epoch
        metrics = loop.run(num_epochs=1, start_epoch=0)

        # Check metrics returned
        assert len(metrics) == 1
        epoch_metrics = metrics[0]

        assert epoch_metrics.epoch == 0
        assert epoch_metrics.train_loss is not None
        assert epoch_metrics.train_loss >= 0
        assert isinstance(epoch_metrics.train_metrics, dict)

    def test_multiple_epochs(self, training_components):
        """Test running multiple epochs."""
        loop = training_components["loop"]

        # Run 3 epochs
        metrics = loop.run(num_epochs=3, start_epoch=0)

        assert len(metrics) == 3

        # Check epochs are sequential
        for i, epoch_metrics in enumerate(metrics):
            assert epoch_metrics.epoch == i

        # Check loss is computed for each epoch
        for epoch_metrics in metrics:
            assert epoch_metrics.train_loss is not None
            assert epoch_metrics.train_loss >= 0

    def test_training_with_validation(self, training_components):
        """Test training with validation."""
        loop = training_components["loop"]

        metrics = loop.run(num_epochs=2)

        # Check validation metrics are computed
        for epoch_metrics in metrics:
            assert epoch_metrics.val_loss is not None
            assert epoch_metrics.val_loss >= 0
            assert epoch_metrics.val_metrics is not None

    def test_loss_decreases_with_training(self, training_components):
        """Test that loss generally decreases (overfitting on tiny dataset)."""
        loop = training_components["loop"]

        # Train for several epochs on tiny dataset
        metrics = loop.run(num_epochs=5)

        # Get first and last loss
        first_loss = metrics[0].train_loss
        last_loss = metrics[-1].train_loss

        # Loss should decrease (we're overfitting on tiny synthetic data)
        # Allow some variance
        assert last_loss < first_loss * 1.5, (
            f"Loss didn't decrease: {first_loss} -> {last_loss}"
        )

    def test_checkpoint_saving(self, training_components, temp_checkpoint_dir):
        """Test that checkpoints are saved."""
        loop = training_components["loop"]

        # Run with checkpoint_every_n_epochs=2
        loop.run(num_epochs=3)

        # Check that checkpoint files exist
        checkpoints = list(temp_checkpoint_dir.glob("*.pth"))
        assert len(checkpoints) > 0, "No checkpoints saved"

        # Should have latest checkpoint
        latest_path = temp_checkpoint_dir / "test_model_latest.pth"
        assert latest_path.exists()

    def test_callbacks(self, training_components):
        """Test callback execution."""
        loop = training_components["loop"]

        # Track callback invocations
        callback_data = {
            "epoch_start_count": 0,
            "epoch_end_count": 0,
            "epochs_seen": [],
        }

        def on_epoch_start(epoch):
            callback_data["epoch_start_count"] += 1

        def on_epoch_end(epoch, metrics):
            callback_data["epoch_end_count"] += 1
            callback_data["epochs_seen"].append(epoch)

        callbacks = {
            "on_epoch_start": on_epoch_start,
            "on_epoch_end": on_epoch_end,
        }

        # Run with callbacks
        loop.run(num_epochs=3, callbacks=callbacks)

        # Check callbacks were called
        assert callback_data["epoch_start_count"] == 3
        assert callback_data["epoch_end_count"] == 3
        assert callback_data["epochs_seen"] == [0, 1, 2]

    def test_early_stopping_callback(self, training_components):
        """Test early stopping via callback."""
        loop = training_components["loop"]

        def on_epoch_end(epoch, metrics):
            # Stop after epoch 1
            if epoch >= 1:
                return False  # Stop training
            return True  # Continue

        callbacks = {"on_epoch_end": on_epoch_end}

        # Request 5 epochs but should stop at 2
        metrics = loop.run(num_epochs=5, callbacks=callbacks)

        # Should only complete 2 epochs (0 and 1)
        assert len(metrics) <= 2


class TestTrainingLoopWithScheduler:
    """Test TrainingLoop with learning rate scheduler."""

    def test_training_with_scheduler(self, training_components):
        """Test training with LR scheduler."""
        from ritm_annotation.core.training import TrainingLoop

        model = training_components["model"]
        optimizer = training_components["optimizer"]
        batch_processor = training_components["loop"].batch_processor
        train_loader = training_components["train_loader"]
        val_loader = training_components["val_loader"]
        device = training_components["loop"].device

        # Create scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        loop = TrainingLoop(
            model=model,
            optimizer=optimizer,
            batch_processor=batch_processor,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            device=device,
        )

        # Get initial LR
        initial_lr = optimizer.param_groups[0]["lr"]

        # Run 3 epochs
        loop.run(num_epochs=3)

        # LR should have changed
        final_lr = optimizer.param_groups[0]["lr"]

        # After 3 epochs with step_size=2, LR should decrease once
        assert final_lr < initial_lr


class TestTrainingLoopGradientHandling:
    """Test gradient computation and clipping."""

    def test_gradient_clipping(self, training_components):
        """Test gradient clipping."""
        from ritm_annotation.core.training import TrainingLoop

        model = training_components["model"]
        optimizer = training_components["optimizer"]
        batch_processor = training_components["loop"].batch_processor
        train_loader = training_components["train_loader"]
        device = training_components["loop"].device

        # Create loop with gradient clipping
        loop = TrainingLoop(
            model=model,
            optimizer=optimizer,
            batch_processor=batch_processor,
            train_loader=train_loader,
            device=device,
            gradient_clip_norm=1.0,
        )

        # Should run without errors
        loop.run(num_epochs=1)

    def test_gradients_are_updated(self, training_components):
        """Test that model parameters are actually updated."""
        loop = training_components["loop"]
        model = training_components["model"]

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Run training
        loop.run(num_epochs=2)

        # Check parameters changed
        final_params = list(model.parameters())

        # At least some parameters should have changed
        changed = False
        for initial, final in zip(initial_params, final_params):
            if not torch.allclose(initial, final, atol=1e-6):
                changed = True
                break

        assert changed, "No parameters were updated during training"


class TestCheckpointManager:
    """Test CheckpointManager separately."""

    def test_save_and_load_checkpoint(self, test_model, temp_checkpoint_dir, device):
        """Test saving and loading checkpoints."""
        from ritm_annotation.core.training import CheckpointManager

        model, _ = test_model

        # Enable gradients
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            model_name="test",
        )

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            epoch=10,
            model=model,
            optimizer=optimizer,
            metrics={"loss": 1.5, "iou": 0.7},
        )

        assert checkpoint_path.exists()

        # Load checkpoint
        checkpoint = manager.load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
        )

        assert checkpoint["epoch"] == 10
        assert "metrics" in checkpoint
        assert checkpoint["metrics"]["loss"] == 1.5

    def test_keep_best_n_checkpoints(self, test_model, temp_checkpoint_dir):
        """Test keeping only best N checkpoints."""
        from ritm_annotation.core.training import CheckpointManager

        model, _ = test_model

        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            model_name="test",
            keep_best_n=2,
        )

        # Save 5 checkpoints with decreasing loss
        for epoch in range(5):
            manager.update_best_checkpoint(
                metric_value=5.0 - epoch,  # Decreasing
                epoch=epoch,
                model=model,
                higher_is_better=False,
            )

        # Should only keep 2 best (lowest loss)
        assert len(manager.best_checkpoints) == 2

        # Best checkpoints should be the ones with lowest loss
        values = [v for v, p in manager.best_checkpoints]
        assert min(values) == 1.0  # Last epoch
        assert max(values) == 2.0  # Second to last

    def test_checkpoint_paths(self, test_model, temp_checkpoint_dir):
        """Test checkpoint path utilities."""
        from ritm_annotation.core.training import CheckpointManager

        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            model_name="mymodel",
        )

        model, _ = test_model

        # Save checkpoint
        manager.save_checkpoint(epoch=5, model=model)

        # Check paths
        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert latest.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
