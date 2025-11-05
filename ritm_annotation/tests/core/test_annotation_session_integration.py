"""
Integration tests for AnnotationSession with real model.

These tests use actual PyTorch models to validate end-to-end behavior.
"""

import pytest
import numpy as np
import torch
from pathlib import Path


pytestmark = pytest.mark.integration


@pytest.fixture
def real_predictor(test_model, device):
    """Create real predictor with test model."""
    from ritm_annotation.inference.predictors import get_predictor

    model, model_cfg = test_model

    predictor = get_predictor(
        model=model,
        device=device,
        net_clicks_limit=None,
        max_size=None,
    )

    return predictor


@pytest.fixture
def annotation_session_with_model(real_predictor):
    """Create AnnotationSession with real model."""
    from ritm_annotation.core.annotation import AnnotationSession

    session = AnnotationSession(real_predictor, prob_thresh=0.5)
    return session


class TestAnnotationSessionWithRealModel:
    """Test AnnotationSession with actual PyTorch model."""

    def test_load_image_real_model(self, annotation_session_with_model, test_image_large):
        """Test loading image with real model."""
        session = annotation_session_with_model

        session.load_image(test_image_large, "test.jpg")

        assert session._image is not None
        assert session.state.image_path == "test.jpg"
        assert session.state.image_shape == test_image_large.shape

    def test_single_click_prediction_real_model(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test prediction from single click with real model."""
        session = annotation_session_with_model
        session.load_image(test_image_large)

        # Add click in center
        h, w = test_image_large.shape[:2]
        prob_map = session.add_click(w // 2, h // 2, is_positive=True)

        # Validate probability map
        assert prob_map is not None
        assert prob_map.shape == (h, w)
        assert prob_map.dtype == np.float32 or prob_map.dtype == np.float64
        assert prob_map.min() >= 0.0
        assert prob_map.max() <= 1.0
        assert not np.isnan(prob_map).any()
        assert not np.isinf(prob_map).any()

        # Should have some activated region
        assert prob_map.max() > 0.1

        # Check state updated
        current_obj = session.state.get_current_object()
        assert len(current_obj.clicks) == 1
        assert current_obj.probability_map is not None
        assert current_obj.mask is not None

    def test_multiple_clicks_refinement(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test iterative refinement with multiple clicks."""
        session = annotation_session_with_model
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]
        cx, cy = w // 2, h // 2

        # First positive click
        prob_map1 = session.add_click(cx, cy, is_positive=True)

        # Second positive click nearby
        prob_map2 = session.add_click(cx + 20, cy + 20, is_positive=True)

        # Predictions should be different
        assert not np.array_equal(prob_map1, prob_map2)

        # Should have 2 clicks
        current_obj = session.state.get_current_object()
        assert len(current_obj.clicks) == 2

    def test_negative_click_refinement(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test negative click refinement."""
        session = annotation_session_with_model
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]
        cx, cy = w // 2, h // 2

        # Positive click
        prob_map1 = session.add_click(cx, cy, is_positive=True)
        max_before = prob_map1.max()

        # Negative click in activated region
        prob_map2 = session.add_click(cx + 10, cy + 10, is_positive=False)

        # Prediction should change
        assert not np.array_equal(prob_map1, prob_map2)

        # Should have 2 clicks
        current_obj = session.state.get_current_object()
        assert len(current_obj.clicks) == 2
        assert current_obj.clicks[0].is_positive == True
        assert current_obj.clicks[1].is_positive == False

    def test_undo_with_real_model(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test undo functionality with real model."""
        session = annotation_session_with_model
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        # Add two clicks
        prob_map1 = session.add_click(w // 2, h // 2, is_positive=True)
        prob_map2 = session.add_click(w // 2 + 20, h // 2 + 20, is_positive=True)

        assert len(session.state.get_current_object().clicks) == 2

        # Undo last click
        success = session.undo_click()
        assert success == True
        assert len(session.state.get_current_object().clicks) == 1

        # Prediction should be restored to first click
        current_prob = session.get_current_prediction()
        # Note: may not be exactly equal due to numerical precision
        assert current_prob is not None

    def test_finish_object_real_model(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test finishing object with real model."""
        session = annotation_session_with_model
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        # Annotate first object
        session.add_click(w // 4, h // 4, is_positive=True)
        session.finish_object()

        # Check result mask created
        result_mask = session.get_result_mask()
        assert result_mask is not None
        assert result_mask.shape == (h, w)
        assert result_mask.max() == 1  # One object

        # Object ID should increment
        assert session.state.current_object_id == 1

    def test_multi_object_annotation_real_model(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test annotating multiple objects with real model."""
        session = annotation_session_with_model
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        # First object
        session.add_click(w // 4, h // 4, is_positive=True)
        session.finish_object()

        # Second object
        session.add_click(3 * w // 4, 3 * h // 4, is_positive=True)
        session.finish_object()

        # Check result
        result_mask = session.get_result_mask()
        assert result_mask is not None
        unique_ids = np.unique(result_mask)
        assert 0 in unique_ids  # Background
        assert 1 in unique_ids  # First object
        assert 2 in unique_ids  # Second object

    def test_visualization_data_real_model(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test getting visualization data with real model."""
        session = annotation_session_with_model
        session.load_image(test_image_large)

        # Add clicks
        h, w = test_image_large.shape[:2]
        session.add_click(w // 2, h // 2, is_positive=True)

        # Get visualization data
        viz_data = session.get_visualization_data()

        assert 'image' in viz_data
        assert 'clicks' in viz_data
        assert 'current_prob_map' in viz_data
        assert 'current_mask' in viz_data

        assert viz_data['image'] is not None
        assert len(viz_data['clicks']) == 1
        assert viz_data['current_prob_map'] is not None

    def test_reset_clicks_real_model(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test resetting clicks with real model."""
        session = annotation_session_with_model
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        # Add multiple clicks
        session.add_click(w // 2, h // 2, is_positive=True)
        session.add_click(w // 2 + 20, h // 2, is_positive=True)

        assert len(session.state.get_current_object().clicks) == 2

        # Reset
        session.reset_clicks()

        current_obj = session.state.get_current_object()
        assert len(current_obj.clicks) == 0
        assert current_obj.probability_map is None
        assert current_obj.mask is None


class TestAnnotationSessionPerformance:
    """Test performance characteristics."""

    def test_prediction_is_fast(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test that prediction completes in reasonable time."""
        import time

        session = annotation_session_with_model
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        start = time.time()
        session.add_click(w // 2, h // 2, is_positive=True)
        elapsed = time.time() - start

        # Should complete in under 5 seconds on CPU
        assert elapsed < 5.0

    def test_multiple_clicks_performance(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test performance with multiple clicks."""
        import time

        session = annotation_session_with_model
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        start = time.time()
        # Add 5 clicks
        for i in range(5):
            session.add_click(
                w // 2 + i * 10,
                h // 2 + i * 10,
                is_positive=(i % 2 == 0)
            )
        elapsed = time.time() - start

        # Should complete in under 10 seconds total
        assert elapsed < 10.0


class TestAnnotationSessionEdgeCases:
    """Test edge cases with real model."""

    def test_click_at_image_boundary(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test clicking at image edges."""
        session = annotation_session_with_model
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        # Click at corners
        prob_map = session.add_click(0, 0, is_positive=True)
        assert prob_map is not None

        session.reset_clicks()
        prob_map = session.add_click(w - 1, h - 1, is_positive=True)
        assert prob_map is not None

    def test_many_clicks_on_same_object(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test adding many clicks to same object."""
        session = annotation_session_with_model
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        # Add 20 clicks
        for i in range(20):
            x = (w // 2 + (i - 10) * 5) % w
            y = (h // 2 + (i - 10) * 5) % h
            prob_map = session.add_click(x, y, is_positive=(i % 3 != 0))
            assert prob_map is not None

        # Check all clicks recorded
        current_obj = session.state.get_current_object()
        assert len(current_obj.clicks) == 20

    def test_finish_without_clicks(
        self,
        annotation_session_with_model,
        test_image_large
    ):
        """Test finishing object without any clicks."""
        session = annotation_session_with_model
        session.load_image(test_image_large)

        # Finish without clicks
        session.finish_object()

        # Should handle gracefully
        result_mask = session.get_result_mask()
        # Result may be None or empty
        if result_mask is not None:
            assert result_mask.sum() == 0  # No pixels marked


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
