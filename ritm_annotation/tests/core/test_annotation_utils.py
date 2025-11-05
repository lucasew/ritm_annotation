"""
Tests for pure annotation utility functions.

These tests validate individual pure functions that have no side effects.
"""

import pytest
import numpy as np
from ritm_annotation.core.annotation.utils import (
    apply_mask_threshold,
    merge_mask_with_result,
    compute_iou,
    validate_image,
    validate_mask,
    compute_click_statistics,
    estimate_object_center,
    find_boundary_points,
    blend_image_with_mask,
    draw_clicks_on_image,
)
from ritm_annotation.core.annotation.state import Click


class TestPureFunctions:
    """Tests for pure utility functions."""

    def test_apply_mask_threshold(self):
        """Test probability thresholding."""
        prob_map = np.array(
            [
                [0.1, 0.4, 0.6],
                [0.3, 0.7, 0.9],
                [0.2, 0.5, 0.8],
            ]
        )

        # Test with default threshold
        binary = apply_mask_threshold(prob_map, 0.5)
        expected = np.array(
            [
                [0, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(binary, expected)

        # Test with different threshold
        binary = apply_mask_threshold(prob_map, 0.3)
        expected = np.array(
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(binary, expected)

    def test_merge_mask_with_result(self):
        """Test merging masks."""
        # Empty result mask
        result = np.zeros((5, 5), dtype=np.int32)

        # First object
        mask1 = np.zeros((5, 5), dtype=np.uint8)
        mask1[1:3, 1:3] = 1

        result = merge_mask_with_result(result, mask1, object_id=0)
        assert result[1, 1] == 1
        assert result[0, 0] == 0

        # Second object
        mask2 = np.zeros((5, 5), dtype=np.uint8)
        mask2[3:5, 3:5] = 1

        result = merge_mask_with_result(result, mask2, object_id=1)
        assert result[1, 1] == 1  # First object unchanged
        assert result[3, 3] == 2  # Second object
        assert result[0, 0] == 0  # Background

    def test_compute_iou(self):
        """Test IoU computation."""
        # Perfect overlap
        mask1 = np.array([[1, 1], [0, 0]])
        mask2 = np.array([[1, 1], [0, 0]])
        assert compute_iou(mask1, mask2) == 1.0

        # No overlap
        mask1 = np.array([[1, 1], [0, 0]])
        mask2 = np.array([[0, 0], [1, 1]])
        assert compute_iou(mask1, mask2) == 0.0

        # Partial overlap
        mask1 = np.array([[1, 1], [1, 0]])
        mask2 = np.array([[1, 0], [1, 1]])
        # Intersection: 2 pixels, Union: 4 pixels
        assert abs(compute_iou(mask1, mask2) - 2 / 4) < 1e-6

        # Empty masks
        mask1 = np.zeros((3, 3))
        mask2 = np.zeros((3, 3))
        assert compute_iou(mask1, mask2) == 0.0

    def test_validate_image(self):
        """Test image validation."""
        # Valid image
        valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
        validate_image(valid_image)  # Should not raise

        # Invalid: None
        with pytest.raises(ValueError, match="None"):
            validate_image(None)

        # Invalid: wrong dimensions
        with pytest.raises(ValueError, match="3D"):
            validate_image(np.zeros((100, 100)))

        # Invalid: wrong number of channels
        with pytest.raises(ValueError, match="3 channels"):
            validate_image(np.zeros((100, 100, 4)))

    def test_validate_mask(self):
        """Test mask validation."""
        image_shape = (100, 100)

        # Valid mask
        valid_mask = np.zeros(image_shape, dtype=np.int32)
        validate_mask(valid_mask, image_shape)  # Should not raise

        # Invalid: None
        with pytest.raises(ValueError, match="None"):
            validate_mask(None, image_shape)

        # Invalid: wrong dimensions
        with pytest.raises(ValueError, match="2D"):
            validate_mask(np.zeros((100, 100, 1)), image_shape)

        # Invalid: wrong shape
        with pytest.raises(ValueError, match="doesn't match"):
            validate_mask(np.zeros((50, 50)), image_shape)

    def test_compute_click_statistics(self):
        """Test click statistics computation."""
        # Empty clicks
        stats = compute_click_statistics([])
        assert stats["num_total"] == 0
        assert stats["num_positive"] == 0
        assert stats["num_negative"] == 0
        assert stats["ratio_positive"] == 0.0

        # Mixed clicks
        clicks = [
            Click(10, 10, True, 0),
            Click(20, 20, True, 0),
            Click(30, 30, False, 0),
        ]
        stats = compute_click_statistics(clicks)
        assert stats["num_total"] == 3
        assert stats["num_positive"] == 2
        assert stats["num_negative"] == 1
        assert abs(stats["ratio_positive"] - 2 / 3) < 1e-6

        # Only positive
        clicks = [Click(10, 10, True, 0), Click(20, 20, True, 0)]
        stats = compute_click_statistics(clicks)
        assert stats["num_positive"] == 2
        assert stats["num_negative"] == 0
        assert stats["ratio_positive"] == 1.0

    def test_estimate_object_center(self):
        """Test object center estimation."""
        # Simple square mask
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1

        center = estimate_object_center(mask)
        assert center is not None
        # Center should be around (5, 5)
        assert abs(center[0] - 5) <= 1
        assert abs(center[1] - 5) <= 1

        # Empty mask
        empty_mask = np.zeros((10, 10), dtype=np.uint8)
        center = estimate_object_center(empty_mask)
        assert center is None

        # Asymmetric mask
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:4, 6:9] = 1
        center = estimate_object_center(mask)
        assert center is not None
        # Should be around (7, 3)
        assert 6 <= center[0] <= 8
        assert 2 <= center[1] <= 4

    def test_find_boundary_points(self):
        """Test finding boundary points."""
        # Circle mask
        mask = np.zeros((50, 50), dtype=np.uint8)
        y, x = np.ogrid[:50, :50]
        circle = (x - 25) ** 2 + (y - 25) ** 2 <= 15**2
        mask[circle] = 1

        # Find boundary points
        points = find_boundary_points(mask, num_points=8)
        assert len(points) == 8

        # All points should be on or near boundary
        for px, py in points:
            # Check point is in mask region
            dist = np.sqrt((px - 25) ** 2 + (py - 25) ** 2)
            assert 10 <= dist <= 20  # Roughly on circle boundary

        # Empty mask
        empty_mask = np.zeros((50, 50), dtype=np.uint8)
        points = find_boundary_points(empty_mask)
        assert points == []

    def test_blend_image_with_mask(self):
        """Test image blending with mask."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 100

        # Create mask with two objects
        mask = np.zeros((50, 50), dtype=np.int32)
        mask[10:20, 10:20] = 1
        mask[30:40, 30:40] = 2

        # Blend
        blended = blend_image_with_mask(image, mask, alpha=0.5)

        # Check shape
        assert blended.shape == image.shape

        # Check that masked regions are different
        assert not np.array_equal(blended[15, 15], image[15, 15])
        assert not np.array_equal(blended[35, 35], image[35, 35])

        # Check background unchanged
        np.testing.assert_array_equal(blended[0, 0], image[0, 0])

        # Empty mask should return original
        empty_mask = np.zeros((50, 50), dtype=np.int32)
        blended = blend_image_with_mask(image, empty_mask, alpha=0.5)
        np.testing.assert_array_equal(blended, image)

    def test_draw_clicks_on_image(self):
        """Test drawing clicks."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 128

        clicks = [
            Click(10, 10, True, 0),
            Click(40, 40, False, 0),
        ]

        result = draw_clicks_on_image(image, clicks, click_radius=3)

        # Check shape unchanged
        assert result.shape == image.shape

        # Check positive click is green-ish
        assert result[10, 10, 1] > 200  # Green channel
        assert result[10, 10, 0] < 50  # Red channel low

        # Check negative click is red-ish
        assert result[40, 40, 0] > 200  # Red channel
        assert result[40, 40, 1] < 50  # Green channel low

        # Check that original image is not modified
        assert not np.array_equal(result[10, 10], image[10, 10])

        # Empty clicks should return copy
        result = draw_clicks_on_image(image, [], click_radius=3)
        np.testing.assert_array_equal(result, image)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_apply_mask_threshold_edge_values(self):
        """Test thresholding at exact boundaries."""
        prob_map = np.array([0.0, 0.5, 1.0])

        # At threshold
        binary = apply_mask_threshold(prob_map, 0.5)
        np.testing.assert_array_equal(binary, [0, 0, 1])

        # Extreme thresholds
        binary = apply_mask_threshold(prob_map, 0.0)
        np.testing.assert_array_equal(binary, [0, 1, 1])

        binary = apply_mask_threshold(prob_map, 1.0)
        np.testing.assert_array_equal(binary, [0, 0, 0])

    def test_compute_iou_single_pixel(self):
        """Test IoU with single pixel masks."""
        mask1 = np.array([[1, 0], [0, 0]])
        mask2 = np.array([[1, 0], [0, 0]])
        assert compute_iou(mask1, mask2) == 1.0

        mask1 = np.array([[1, 0], [0, 0]])
        mask2 = np.array([[0, 1], [0, 0]])
        assert compute_iou(mask1, mask2) == 0.0

    def test_find_boundary_points_small_mask(self):
        """Test boundary finding with very small mask."""
        # 3x3 square
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[4:7, 4:7] = 1

        points = find_boundary_points(mask, num_points=20)
        # Should return all available boundary points
        assert len(points) > 0
        assert len(points) <= 12  # Perimeter of 3x3 square

    def test_merge_mask_overlapping_objects(self):
        """Test merging when objects overlap."""
        result = np.zeros((10, 10), dtype=np.int32)

        # First object
        mask1 = np.zeros((10, 10), dtype=np.uint8)
        mask1[2:6, 2:6] = 1
        result = merge_mask_with_result(result, mask1, object_id=0)

        # Overlapping second object
        mask2 = np.zeros((10, 10), dtype=np.uint8)
        mask2[4:8, 4:8] = 1
        result = merge_mask_with_result(result, mask2, object_id=1)

        # Overlapping region should belong to second object
        assert result[5, 5] == 2
        # Non-overlapping first object region should remain
        assert result[3, 3] == 1


class TestReproducibility:
    """Test that functions are deterministic."""

    def test_compute_iou_reproducible(self):
        """Test that IoU is deterministic."""
        mask1 = np.random.randint(0, 2, (100, 100))
        mask2 = np.random.randint(0, 2, (100, 100))

        iou1 = compute_iou(mask1, mask2)
        iou2 = compute_iou(mask1, mask2)
        assert iou1 == iou2

    def test_estimate_center_reproducible(self):
        """Test that center estimation is deterministic."""
        mask = np.random.randint(0, 2, (50, 50)).astype(np.uint8)

        center1 = estimate_object_center(mask)
        center2 = estimate_object_center(mask)
        assert center1 == center2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
