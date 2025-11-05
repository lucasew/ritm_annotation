"""
Tests for AnnotationSession.

These tests demonstrate how the new modular architecture
enables easy unit testing without GUI dependencies.
"""

import pytest
import numpy as np
from unittest.mock import Mock


@pytest.fixture
def mock_predictor():
    """Create a mock predictor for testing."""
    predictor = Mock()
    predictor.set_input_image = Mock()
    predictor.get_prediction = Mock(return_value=np.random.rand(100, 100))
    predictor.get_states = Mock(return_value={})
    predictor.set_state = Mock()
    return predictor


@pytest.fixture
def test_image():
    """Create a test image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def annotation_session(mock_predictor):
    """Create an AnnotationSession with mock predictor."""
    from ritm_annotation.core.annotation import AnnotationSession

    return AnnotationSession(mock_predictor, prob_thresh=0.5)


class TestAnnotationSession:
    """Test suite for AnnotationSession."""

    def test_initialization(self, annotation_session):
        """Test session initialization."""
        assert annotation_session.prob_thresh == 0.5
        assert annotation_session.state is not None
        assert annotation_session.events is not None

    def test_load_image(self, annotation_session, test_image, mock_predictor):
        """Test image loading."""
        annotation_session.load_image(test_image, "test.jpg")

        assert annotation_session._image is not None
        assert annotation_session.state.image_path == "test.jpg"
        assert annotation_session.state.image_shape == test_image.shape
        mock_predictor.set_input_image.assert_called_once()

    def test_add_click(self, annotation_session, test_image):
        """Test adding clicks."""
        annotation_session.load_image(test_image)

        # Add first click
        prob_map = annotation_session.add_click(50, 50, is_positive=True)

        current_obj = annotation_session.state.get_current_object()
        assert len(current_obj.clicks) == 1
        assert current_obj.clicks[0].x == 50
        assert current_obj.clicks[0].y == 50
        assert current_obj.clicks[0].is_positive
        assert prob_map is not None

        # Add second click
        annotation_session.add_click(70, 70, is_positive=False)
        assert len(current_obj.clicks) == 2

    def test_undo_click(self, annotation_session, test_image):
        """Test undo functionality."""
        annotation_session.load_image(test_image)

        # Add clicks
        annotation_session.add_click(50, 50, is_positive=True)
        annotation_session.add_click(70, 70, is_positive=True)

        current_obj = annotation_session.state.get_current_object()
        assert len(current_obj.clicks) == 2

        # Undo last click
        result = annotation_session.undo_click()
        assert result
        assert len(current_obj.clicks) == 1

        # Undo first click
        result = annotation_session.undo_click()
        assert result
        assert len(current_obj.clicks) == 0

        # Try to undo when no history
        result = annotation_session.undo_click()
        assert not result

    def test_reset_clicks(self, annotation_session, test_image):
        """Test resetting clicks."""
        annotation_session.load_image(test_image)

        # Add clicks
        annotation_session.add_click(50, 50, is_positive=True)
        annotation_session.add_click(70, 70, is_positive=True)

        # Reset
        annotation_session.reset_clicks()

        current_obj = annotation_session.state.get_current_object()
        assert len(current_obj.clicks) == 0
        assert current_obj.probability_map is None
        assert current_obj.mask is None

    def test_finish_object(self, annotation_session, test_image):
        """Test finishing an object."""
        annotation_session.load_image(test_image)

        # Add click and finish
        annotation_session.add_click(50, 50, is_positive=True)

        initial_obj_id = annotation_session.state.current_object_id
        annotation_session.finish_object()

        # Check object was finished
        finished_obj = annotation_session.state.objects[0]
        assert finished_obj.is_finished
        assert finished_obj.object_id == initial_obj_id

        # Check new object was created
        assert annotation_session.state.current_object_id == initial_obj_id + 1

    def test_multi_object_annotation(self, annotation_session, test_image):
        """Test annotating multiple objects."""
        annotation_session.load_image(test_image)

        # First object
        annotation_session.add_click(30, 30, is_positive=True)
        annotation_session.finish_object()

        # Second object
        annotation_session.add_click(70, 70, is_positive=True)
        annotation_session.finish_object()

        # Check result
        assert len(annotation_session.state.objects) >= 2
        finished_objects = [
            obj for obj in annotation_session.state.objects if obj.is_finished
        ]
        assert len(finished_objects) == 2

    def test_load_mask(self, annotation_session, test_image):
        """Test loading existing mask."""
        annotation_session.load_image(test_image)

        # Create test mask with 2 objects
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[20:40, 20:40] = 1  # Object 1
        mask[60:80, 60:80] = 2  # Object 2

        annotation_session.load_mask(mask)

        # Check objects were created
        assert annotation_session.state.result_mask is not None
        assert len(annotation_session.state.objects) == 2

    def test_get_visualization_data(self, annotation_session, test_image):
        """Test getting visualization data."""
        annotation_session.load_image(test_image)
        annotation_session.add_click(50, 50, is_positive=True)

        viz_data = annotation_session.get_visualization_data()

        assert "image" in viz_data
        assert "clicks" in viz_data
        assert "current_object_id" in viz_data
        assert len(viz_data["clicks"]) == 1

    def test_event_emission(self, annotation_session, test_image):
        """Test that events are emitted correctly."""
        from ritm_annotation.core.annotation import EventType

        # Track events
        events_received = []

        def on_event(event):
            events_received.append(event.event_type)

        # Subscribe to events
        annotation_session.events.on(EventType.IMAGE_LOADED, on_event)
        annotation_session.events.on(EventType.CLICK_ADDED, on_event)
        annotation_session.events.on(EventType.PREDICTION_COMPLETED, on_event)

        # Perform actions
        annotation_session.load_image(test_image)
        annotation_session.add_click(50, 50, is_positive=True)

        # Check events were received
        assert EventType.IMAGE_LOADED in events_received
        assert EventType.CLICK_ADDED in events_received
        assert EventType.PREDICTION_COMPLETED in events_received


class TestAnnotationState:
    """Test suite for AnnotationState."""

    def test_state_serialization(self):
        """Test state can be serialized to dict."""
        from ritm_annotation.core.annotation import AnnotationState, ObjectState, Click

        state = AnnotationState(
            image_path="test.jpg",
            image_shape=(100, 100, 3),
        )

        obj = ObjectState(object_id=0)
        obj.add_click(Click(x=50, y=50, is_positive=True))
        state.objects.append(obj)

        # Serialize
        state_dict = state.to_dict()

        assert state_dict["image_path"] == "test.jpg"
        assert state_dict["image_shape"] == (100, 100, 3)
        assert len(state_dict["objects"]) == 1


class TestEventSystem:
    """Test suite for event system."""

    def test_event_subscription(self):
        """Test subscribing to events."""
        from ritm_annotation.core.annotation import (
            EventEmitter,
            EventType,
            AnnotationEvent,
        )

        emitter = EventEmitter()
        events_received = []

        def callback(event):
            events_received.append(event)

        emitter.on(EventType.CLICK_ADDED, callback)
        emitter.emit(AnnotationEvent(EventType.CLICK_ADDED, {"x": 50}))

        assert len(events_received) == 1
        assert events_received[0].event_type == EventType.CLICK_ADDED

    def test_event_unsubscription(self):
        """Test unsubscribing from events."""
        from ritm_annotation.core.annotation import (
            EventEmitter,
            EventType,
            AnnotationEvent,
        )

        emitter = EventEmitter()
        events_received = []

        def callback(event):
            events_received.append(event)

        emitter.on(EventType.CLICK_ADDED, callback)
        emitter.emit(AnnotationEvent(EventType.CLICK_ADDED))
        assert len(events_received) == 1

        # Unsubscribe
        emitter.off(EventType.CLICK_ADDED, callback)
        emitter.emit(AnnotationEvent(EventType.CLICK_ADDED))
        assert len(events_received) == 1  # No new event

    def test_multiple_listeners(self):
        """Test multiple listeners for same event."""
        from ritm_annotation.core.annotation import (
            EventEmitter,
            EventType,
            AnnotationEvent,
        )

        emitter = EventEmitter()
        counter1 = [0]
        counter2 = [0]

        def callback1(event):
            counter1[0] += 1

        def callback2(event):
            counter2[0] += 1

        emitter.on(EventType.CLICK_ADDED, callback1)
        emitter.on(EventType.CLICK_ADDED, callback2)

        emitter.emit(AnnotationEvent(EventType.CLICK_ADDED))

        assert counter1[0] == 1
        assert counter2[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
