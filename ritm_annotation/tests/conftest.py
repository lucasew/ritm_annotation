"""
Test fixtures and utilities for RITM annotation tests.

Provides reusable fixtures for models, datasets, and test data.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock


@pytest.fixture(scope="session")
def device():
    """Get device for testing (prefer CPU for consistent tests)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def test_model(device):
    """
    Create a small test model for testing.

    Uses HRNet18 which is small enough for CI but real enough to test properly.
    """
    from ritm_annotation.models.iter_mask.hrnet18_cocolvis_itermask_3p import init_model
    from easydict import EasyDict as edict

    cfg = edict()
    cfg.device = device

    # Create model without pretrained weights (dry_run=True)
    model, model_cfg = init_model(cfg, dry_run=True)
    model.eval()

    return model, model_cfg


@pytest.fixture
def test_image():
    """Create a test RGB image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def test_image_large():
    """Create a larger test image for more realistic tests."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_mask():
    """Create a test segmentation mask."""
    mask = np.zeros((100, 100), dtype=np.int32)
    # Object 1: circle at (30, 30)
    y, x = np.ogrid[:100, :100]
    circle1 = (x - 30) ** 2 + (y - 30) ** 2 <= 15**2
    mask[circle1] = 1

    # Object 2: circle at (70, 70)
    circle2 = (x - 70) ** 2 + (y - 70) ** 2 <= 10**2
    mask[circle2] = 2

    return mask


@pytest.fixture
def test_batch(device):
    """Create a test batch for training."""
    batch_size = 2
    batch = {
        "images": torch.randn(batch_size, 4, 320, 480).to(device),  # 4 channels: RGB + prev_mask
        "points": torch.zeros(batch_size, 2, 3).to(device),  # (batch, num_max_points*2, 3)
        "instances": torch.randint(0, 2, (batch_size, 1, 320, 480)).float().to(device),
    }
    # Set first point as valid (y, x, index) = (50, 50, 0)
    batch["points"][:, 0] = torch.tensor([50.0, 50.0, 0.0])
    # Second point as padding
    batch["points"][:, 1] = torch.tensor([-1.0, -1.0, -1.0])
    return batch


@pytest.fixture
def mock_predictor():
    """Create a mock predictor that returns realistic outputs."""
    predictor = Mock()

    def mock_set_image(image):
        predictor._image_shape = (
            image.shape[:2] if hasattr(image, "shape") else (100, 100)
        )

    def mock_get_prediction(clicker):
        # Return a circular mask around first positive click
        clicks = clicker.get_clicks()
        if not clicks:
            return np.zeros(predictor._image_shape, dtype=np.float32)

        # Get first positive click
        pos_clicks = [c for c in clicks if c.is_positive]
        if not pos_clicks:
            return np.zeros(predictor._image_shape, dtype=np.float32)

        # Create circular mask
        h, w = predictor._image_shape
        y, x = np.ogrid[:h, :w]
        cx, cy = pos_clicks[0].coords

        mask = np.zeros((h, w), dtype=np.float32)
        circle = (x - cx) ** 2 + (y - cy) ** 2 <= 20**2
        mask[circle] = 1.0

        # Add noise from negative clicks
        for click in clicks:
            if not click.is_positive:
                cx, cy = click.coords
                neg_circle = (x - cx) ** 2 + (y - cy) ** 2 <= 15**2
                mask[neg_circle] = 0.0

        return mask

    predictor.set_input_image = mock_set_image
    predictor.get_prediction = mock_get_prediction
    predictor.get_states = Mock(return_value={})
    predictor.set_state = Mock()
    predictor._image_shape = (100, 100)

    return predictor


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_loss_fn():
    """Create simple loss function for testing."""
    from ritm_annotation.model.losses import (
        NormalizedFocalLossSigmoid,
        SigmoidBinaryCrossEntropyLoss,
    )

    return {
        "instance_loss": NormalizedFocalLossSigmoid(alpha=0.5, gamma=2),
        "instance_aux_loss": SigmoidBinaryCrossEntropyLoss(),
    }


@pytest.fixture
def simple_metric():
    """Create simple metric for testing."""
    from ritm_annotation.model.metrics import AdaptiveIoU

    metric = AdaptiveIoU()
    return metric


def create_synthetic_dataset(num_samples=10, image_size=(320, 480)):
    """
    Create synthetic dataset for testing training loop.

    Returns torch Dataset with realistic structure.
    """

    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, image_size):
            self.num_samples = num_samples
            self.image_size = image_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            h, w = self.image_size

            # Random image with 4 channels (RGB + prev_mask)
            image = torch.randn(4, h, w)

            # Random mask
            mask = torch.zeros(1, h, w)
            # Add random circle
            cx, cy = (
                np.random.randint(w // 4, 3 * w // 4),
                np.random.randint(h // 4, 3 * h // 4),
            )
            radius = np.random.randint(10, 30)
            y, x = np.ogrid[:h, :w]
            circle = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
            mask[0, circle] = 1.0

            # Random points (1 positive point in the mask)
            # Format: (num_max_points * 2, 3) where first half is positive clicks, second half is negative clicks
            points = torch.zeros(2, 3)  # 1 max point * 2 (pos + neg)
            if circle.sum() > 0:
                valid_points = np.argwhere(circle)
                idx = np.random.randint(len(valid_points))
                py, px = valid_points[idx]
                points[0] = torch.tensor([py, px, 0.0])  # (y, x, index)  # First click
            else:
                points[0] = torch.tensor([-1, -1, -1])  # Padding
            points[1] = torch.tensor([-1, -1, -1])  # No negative clicks (padding)

            return {
                "images": image,
                "points": points,
                "instances": mask,
            }

    return SyntheticDataset(num_samples, image_size)


def assert_tensor_equal(t1, t2, rtol=1e-5, atol=1e-8):
    """Assert two tensors are equal within tolerance."""
    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        assert torch.allclose(t1, t2, rtol=rtol, atol=atol), (
            f"Tensors not equal: max diff = {(t1 - t2).abs().max()}"
        )
    elif isinstance(t1, np.ndarray) and isinstance(t2, np.ndarray):
        assert np.allclose(t1, t2, rtol=rtol, atol=atol), (
            f"Arrays not equal: max diff = {np.abs(t1 - t2).max()}"
        )
    else:
        raise TypeError(f"Unsupported types: {type(t1)}, {type(t2)}")


def assert_valid_probability_map(prob_map):
    """Assert probability map is valid."""
    assert prob_map is not None, "Probability map is None"
    assert isinstance(prob_map, (np.ndarray, torch.Tensor)), (
        f"Invalid type: {type(prob_map)}"
    )

    if isinstance(prob_map, torch.Tensor):
        prob_map = prob_map.cpu().numpy()

    assert prob_map.min() >= 0.0, f"Min probability < 0: {prob_map.min()}"
    assert prob_map.max() <= 1.0, f"Max probability > 1: {prob_map.max()}"
    assert not np.isnan(prob_map).any(), "Probability map contains NaN"
    assert not np.isinf(prob_map).any(), "Probability map contains Inf"
