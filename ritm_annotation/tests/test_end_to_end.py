"""
End-to-end integration tests.

Tests complete workflows from start to finish.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import cv2


pytestmark = pytest.mark.integration


class TestAnnotationWorkflow:
    """Test complete annotation workflow."""

    def test_annotate_save_and_load(self, test_model, test_image_large, device):
        """Test complete annotation workflow with save/load."""
        from ritm_annotation.core.annotation import AnnotationSession
        from ritm_annotation.inference.predictors import get_predictor

        model, model_cfg = test_model
        predictor = get_predictor(model, device=device)

        # Create session
        session = AnnotationSession(predictor, prob_thresh=0.5)
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        # Annotate first object
        session.add_click(w // 4, h // 4, is_positive=True)
        session.add_click(w // 4 + 10, h // 4, is_positive=True)
        session.finish_object()

        # Annotate second object
        session.add_click(3 * w // 4, 3 * h // 4, is_positive=True)
        session.finish_object()

        # Get result
        result_mask = session.get_result_mask()
        assert result_mask is not None

        # Save mask
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = Path(tmpdir) / "mask.png"
            cv2.imwrite(str(mask_path), result_mask.astype(np.uint8))

            # Load mask back
            loaded_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

            # Create new session and load mask
            new_session = AnnotationSession(predictor, prob_thresh=0.5)
            new_session.load_image(test_image_large)
            new_session.load_mask(loaded_mask)

            # Check mask was loaded correctly
            assert new_session.state.result_mask is not None
            assert new_session.state.result_mask.shape == result_mask.shape

    def test_interactive_refinement_workflow(
        self, test_model, test_image_large, device
    ):
        """Test interactive refinement workflow."""
        from ritm_annotation.core.annotation import AnnotationSession
        from ritm_annotation.inference.predictors import get_predictor
        from ritm_annotation.core.annotation.utils import compute_iou

        model, _ = test_model
        predictor = get_predictor(model, device=device)

        session = AnnotationSession(predictor, prob_thresh=0.5)
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        # Initial positive click
        prob_map1 = session.add_click(w // 2, h // 2, is_positive=True)
        mask1 = (prob_map1 > 0.5).astype(np.uint8)

        # Add another positive click
        prob_map2 = session.add_click(w // 2 + 20, h // 2, is_positive=True)
        mask2 = (prob_map2 > 0.5).astype(np.uint8)

        # Masks should be similar but refined
        iou = compute_iou(mask1, mask2)
        assert iou > 0.3, "Masks too different after refinement"

        # Add negative click
        prob_map3 = session.add_click(w // 2 + 40, h // 2, is_positive=False)
        mask3 = (prob_map3 > 0.5).astype(np.uint8)

        # Mask should change
        assert not np.array_equal(mask2, mask3)

    def test_undo_redo_workflow(self, test_model, test_image_large, device):
        """Test undo/redo workflow."""
        from ritm_annotation.core.annotation import AnnotationSession
        from ritm_annotation.inference.predictors import get_predictor

        model, _ = test_model
        predictor = get_predictor(model, device=device)

        session = AnnotationSession(predictor, prob_thresh=0.5)
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        # Add clicks and save states
        session.add_click(w // 2, h // 2, is_positive=True)

        session.add_click(w // 2 + 20, h // 2, is_positive=True)

        session.add_click(w // 2 - 20, h // 2, is_positive=False)

        # Undo twice
        session.undo_click()
        session.undo_click()

        # Should be back to state after first click
        # Note: May not be exactly equal due to predictor state, but should have 1 click
        assert len(session.state.get_current_object().clicks) == 1


class TestTrainingWorkflow:
    """Test complete training workflow."""

    def test_train_save_and_resume(self, test_model, device):
        """Test training, saving checkpoint, and resuming."""
        from ritm_annotation.core.training import (
            TrainingLoop,
            BatchProcessor,
            CheckpointManager,
        )
        from ritm_annotation.tests.conftest import create_synthetic_dataset
        from torch.utils.data import DataLoader

        model, _ = test_model

        # Enable gradients
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Create components
        batch_processor = BatchProcessor(
            model=model,
            loss_fn={
                "instance_loss": torch.nn.BCEWithLogitsLoss(),
                "instance_aux_loss": torch.nn.BCEWithLogitsLoss(),
            },
            metrics=[],
            max_interactive_points=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                model_name="test",
            )

            train_dataset = create_synthetic_dataset(10, (320, 480))
            train_loader = DataLoader(train_dataset, batch_size=2)

            loop = TrainingLoop(
                model=model,
                optimizer=optimizer,
                batch_processor=batch_processor,
                train_loader=train_loader,
                checkpoint_manager=checkpoint_manager,
                device=device,
            )

            # Train for 2 epochs
            loop.run(num_epochs=2)

            # Save checkpoint
            checkpoint_path = checkpoint_manager.save_checkpoint(
                epoch=2,
                model=model,
                optimizer=optimizer,
            )

            # Create new model and optimizer
            model2, _ = test_model
            for param in model2.parameters():
                param.requires_grad = True
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)

            # Load checkpoint
            checkpoint_manager.load_checkpoint(
                checkpoint_path,
                model2,
                optimizer2,
            )

            # Continue training
            loop2 = TrainingLoop(
                model=model2,
                optimizer=optimizer2,
                batch_processor=batch_processor,
                train_loader=train_loader,
                checkpoint_manager=checkpoint_manager,
                device=device,
            )

            # Should resume from epoch 3
            metrics2 = loop2.run(num_epochs=4, start_epoch=2)

            # Should train for 2 more epochs
            assert len(metrics2) == 2


class TestModelPersistence:
    """Test model save/load workflows."""

    def test_save_and_load_model_state(self, test_model, device):
        """Test saving and loading model state."""
        model, _ = test_model

        # Get initial output
        dummy_input = {
            "images": torch.randn(1, 3, 320, 480).to(device),
            "points": torch.zeros(1, 1, 3).to(device),
        }

        with torch.no_grad():
            initial_output = model(**dummy_input)

        # Save state
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model_state.pth"
            torch.save(model.state_dict(), save_path)

            # Create new model and load state
            model2, _ = test_model
            model2.load_state_dict(torch.load(save_path))

            # Get output from loaded model
            with torch.no_grad():
                loaded_output = model2(**dummy_input)

            # Outputs should be identical
            for key in initial_output:
                if isinstance(initial_output[key], torch.Tensor):
                    assert torch.allclose(
                        initial_output[key], loaded_output[key], rtol=1e-5, atol=1e-8
                    )


class TestGUIAdapterWorkflow:
    """Test GUI adapter workflow."""

    def test_gui_adapter_complete_workflow(self, test_model, test_image_large, device):
        """Test complete workflow with GUI adapter."""
        from ritm_annotation.core.annotation import AnnotationSession
        from ritm_annotation.interfaces import GUIAnnotationAdapter
        from ritm_annotation.inference.predictors import get_predictor

        model, _ = test_model
        predictor = get_predictor(model, device=device)

        # Create session
        session = AnnotationSession(predictor, prob_thresh=0.5)

        # Track callback invocations
        update_count = [0]

        def update_callback():
            update_count[0] += 1

        # Create adapter
        adapter = GUIAnnotationAdapter(
            session=session,
            update_image_callback=update_callback,
            click_radius=3,
        )

        # Use adapter (compatible API)
        adapter.set_image(test_image_large)
        assert update_count[0] > 0  # Callback should be called

        h, w = test_image_large.shape[:2]

        # Add clicks
        update_count[0] = 0
        adapter.add_click(w // 2, h // 2, is_positive=True)
        assert update_count[0] > 0  # Callback should be called

        # Get visualization
        vis = adapter.get_visualization(alpha_blend=0.5)
        assert vis is not None
        assert vis.shape == test_image_large.shape

        # Finish object
        update_count[0] = 0
        adapter.partially_finish_object()
        assert update_count[0] > 0  # Callback should be called

        # Undo
        adapter.undo_click()

        # Reset
        adapter.reset_clicks()


class TestDataIntegrity:
    """Test data integrity throughout workflows."""

    def test_mask_integrity_through_workflow(
        self, test_model, test_image_large, device
    ):
        """Test that masks remain valid throughout workflow."""
        from ritm_annotation.core.annotation import AnnotationSession
        from ritm_annotation.inference.predictors import get_predictor
        from ritm_annotation.core.annotation.utils import validate_mask

        model, _ = test_model
        predictor = get_predictor(model, device=device)

        session = AnnotationSession(predictor, prob_thresh=0.5)
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        # Add multiple clicks and check mask validity
        for i in range(5):
            x = (w // 2 + i * 10) % w
            y = (h // 2 + i * 10) % h
            prob_map = session.add_click(x, y, is_positive=(i % 2 == 0))

            # Validate probability map
            assert prob_map.shape == (h, w)
            assert prob_map.min() >= 0.0
            assert prob_map.max() <= 1.0

            # Get current mask
            current_mask = session.get_current_prediction()
            assert current_mask.shape == (h, w)

        # Finish and check result mask
        session.finish_object()
        result_mask = session.get_result_mask()

        if result_mask is not None:
            validate_mask(result_mask, (h, w))

    def test_state_consistency_through_workflow(
        self, test_model, test_image_large, device
    ):
        """Test that state remains consistent."""
        from ritm_annotation.core.annotation import AnnotationSession
        from ritm_annotation.inference.predictors import get_predictor

        model, _ = test_model
        predictor = get_predictor(model, device=device)

        session = AnnotationSession(predictor, prob_thresh=0.5)
        session.load_image(test_image_large)

        h, w = test_image_large.shape[:2]

        # Add clicks and check state
        for i in range(3):
            session.add_click(w // 2 + i * 10, h // 2, is_positive=True)

            # Check state consistency
            state = session.state
            current_obj = state.get_current_object()

            assert len(current_obj.clicks) == i + 1
            assert current_obj.probability_map is not None
            assert current_obj.mask is not None

            # Check clicks match
            for j, click in enumerate(current_obj.clicks):
                assert click.object_id == state.current_object_id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
