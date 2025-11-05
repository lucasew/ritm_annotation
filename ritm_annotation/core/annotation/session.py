"""
Annotation session management.

Core logic for managing an interactive annotation session.
UI-agnostic - can be used with any interface (GUI, Web, CLI).
"""

from typing import Optional, Dict, Any
import numpy as np

from .events import EventEmitter, AnnotationEvent, EventType
from .state import AnnotationState, ObjectState, Click


class AnnotationSession:
    """
    Manages the state and logic of an annotation session.

    This class handles:
    - Click management
    - State history for undo/redo
    - Object lifecycle (start, finish)
    - Prediction coordination
    - Event emission for UI updates

    The session is UI-agnostic - it emits events that UI components
    can listen to, rather than directly manipulating UI elements.
    """

    def __init__(self, predictor, prob_thresh: float = 0.5):
        """
        Initialize annotation session.

        Args:
            predictor: Model predictor for generating segmentations
            prob_thresh: Probability threshold for mask generation
        """
        self.predictor = predictor
        self.prob_thresh = prob_thresh

        # Current state
        self.state = AnnotationState()

        # History for undo functionality
        self._state_history: list = []

        # Current image
        self._image: Optional[np.ndarray] = None

        # Event emitter for UI notifications
        self.events = EventEmitter()

        # Cache for predictions
        self._prediction_cache: Dict[str, Any] = {}

    def load_image(self, image: np.ndarray, image_path: Optional[str] = None):
        """
        Load a new image for annotation.

        Args:
            image: RGB image as numpy array
            image_path: Optional path to the image file
        """
        self._image = image
        self.state = AnnotationState(
            image_path=image_path,
            image_shape=image.shape,
        )
        self._state_history.clear()
        self._prediction_cache.clear()

        # Set image in predictor
        self.predictor.set_input_image(image)

        self.events.emit(
            AnnotationEvent(
                EventType.IMAGE_LOADED, {"image_shape": image.shape, "path": image_path}
            )
        )

    def add_click(self, x: float, y: float, is_positive: bool) -> np.ndarray:
        """
        Add a click and get updated prediction.

        Args:
            x: X coordinate
            y: Y coordinate
            is_positive: True for positive click, False for negative

        Returns:
            Updated probability map
        """
        if self._image is None:
            raise ValueError("No image loaded")

        # Save current state for undo
        self._save_state()

        # Create click and add to current object
        click = Click(
            x=x, y=y, is_positive=is_positive, object_id=self.state.current_object_id
        )
        current_obj = self.state.get_current_object()
        current_obj.add_click(click)

        # Emit click event
        self.events.emit(
            AnnotationEvent(
                EventType.CLICK_ADDED,
                {"click": click.to_dict(), "num_clicks": len(current_obj.clicks)},
            )
        )

        # Get prediction
        prob_map = self._get_prediction(current_obj)
        current_obj.probability_map = prob_map
        current_obj.mask = (prob_map > self.prob_thresh).astype(np.uint8)

        self.events.emit(
            AnnotationEvent(
                EventType.PREDICTION_COMPLETED,
                {"object_id": self.state.current_object_id},
            )
        )

        return prob_map

    def undo_click(self) -> bool:
        """
        Undo the last click.

        Returns:
            True if undo was successful, False if no history
        """
        if not self._state_history:
            return False

        # Restore previous state
        prev_state = self._state_history.pop()
        self.state = prev_state["state"]

        # Restore predictor state
        if "predictor_state" in prev_state:
            self.predictor.set_states(prev_state["predictor_state"])

        self.events.emit(AnnotationEvent(EventType.CLICK_UNDONE))
        return True

    def reset_clicks(self):
        """Reset all clicks for the current object."""
        current_obj = self.state.get_current_object()
        current_obj.clicks.clear()
        current_obj.probability_map = None
        current_obj.mask = None

        # Reset predictor
        self.predictor.set_input_image(self._image)
        self._state_history.clear()

        self.events.emit(AnnotationEvent(EventType.CLICKS_RESET))

    def finish_object(self):
        """
        Finish annotating the current object and start a new one.

        Updates the result mask with the current object's mask.
        """
        current_obj = self.state.get_current_object()

        if current_obj.mask is not None:
            # Update result mask
            if self.state.result_mask is None:
                self.state.result_mask = np.zeros(self._image.shape[:2], dtype=np.int32)

            # Add current object to result mask
            object_mask = current_obj.mask > 0
            self.state.result_mask[object_mask] = self.state.current_object_id + 1

            # Mark as finished
            self.state.finish_current_object()

            # Reset predictor for next object
            self.predictor.set_input_image(self._image)
            self._state_history.clear()

            self.events.emit(
                AnnotationEvent(
                    EventType.OBJECT_FINISHED, {"object_id": current_obj.object_id}
                )
            )

            self.events.emit(
                AnnotationEvent(
                    EventType.OBJECT_STARTED,
                    {"object_id": self.state.current_object_id},
                )
            )

    def load_mask(self, mask: np.ndarray):
        """
        Load a pre-existing mask.

        Args:
            mask: Mask array (H, W) with integer labels
        """
        self.state.result_mask = mask.copy()

        # Parse objects from mask
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels > 0]

        self.state.objects.clear()
        for label in unique_labels:
            obj = ObjectState(
                object_id=int(label) - 1,
                mask=(mask == label).astype(np.uint8),
                is_finished=True,
            )
            self.state.objects.append(obj)

        if len(unique_labels) > 0:
            self.state.current_object_id = int(np.max(unique_labels))
        else:
            self.state.current_object_id = 0

        self.events.emit(
            AnnotationEvent(EventType.MASK_LOADED, {"num_objects": len(unique_labels)})
        )

    def get_result_mask(self) -> Optional[np.ndarray]:
        """
        Get the final result mask.

        Returns:
            Mask with all finished objects, or None if no objects
        """
        return self.state.result_mask

    def get_current_prediction(self) -> Optional[np.ndarray]:
        """
        Get the current object's probability map.

        Returns:
            Probability map or None
        """
        current_obj = self.state.get_current_object()
        return current_obj.probability_map

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data needed for visualization.

        Returns:
            Dictionary with visualization data
        """
        current_obj = self.state.get_current_object()

        return {
            "image": self._image,
            "result_mask": self.state.result_mask,
            "current_prob_map": current_obj.probability_map,
            "current_mask": current_obj.mask,
            "clicks": current_obj.clicks,
            "all_clicks": self.state.get_all_clicks(),
            "current_object_id": self.state.current_object_id,
            "num_objects": len([o for o in self.state.objects if o.is_finished]),
        }

    def _get_prediction(self, obj: ObjectState) -> np.ndarray:
        """
        Get prediction from the model for the current object.

        Args:
            obj: Object state with clicks

        Returns:
            Probability map
        """
        if not obj.clicks:
            return np.zeros(self._image.shape[:2], dtype=np.float32)

        # Convert clicks to predictor format
        # The predictor expects a clicker object
        # We'll need to update the predictor with clicks
        from ...inference.clicker import Clicker, Click as InferenceClick

        clicker = Clicker()
        for click in obj.clicks:
            # Convert from state Click (x, y) to inference Click (is_positive, coords=(y, x))
            inference_click = InferenceClick(
                is_positive=click.is_positive, coords=(click.y, click.x)
            )
            clicker.add_click(inference_click)

        # Get prediction
        try:
            pred = self.predictor.get_prediction(clicker)
            return pred
        except Exception as e:
            self.events.emit(
                AnnotationEvent(EventType.PREDICTION_FAILED, {"error": str(e)})
            )
            raise

    def _save_state(self):
        """Save current state to history for undo."""
        # Deep copy the state
        import copy

        state_copy = copy.deepcopy(self.state)

        # Save predictor state if available
        predictor_state = None
        if hasattr(self.predictor, "get_states"):
            predictor_state = self.predictor.get_states()

        self._state_history.append(
            {
                "state": state_copy,
                "predictor_state": predictor_state,
            }
        )

        # Limit history size
        max_history = 100
        if len(self._state_history) > max_history:
            self._state_history.pop(0)
