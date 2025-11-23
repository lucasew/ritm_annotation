"""
Event system for annotation workflow.

Provides a decoupled way for the annotation core to notify UI components
about state changes without depending on specific UI frameworks.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass


class EventType(Enum):
    """Types of events that can occur during annotation."""

    # Image events
    IMAGE_LOADED = "image_loaded"
    IMAGE_CHANGED = "image_changed"

    # Click events
    CLICK_ADDED = "click_added"
    CLICK_UNDONE = "click_undone"
    CLICKS_RESET = "clicks_reset"

    # Prediction events
    PREDICTION_STARTED = "prediction_started"
    PREDICTION_COMPLETED = "prediction_completed"
    PREDICTION_FAILED = "prediction_failed"

    # Object events
    OBJECT_FINISHED = "object_finished"
    OBJECT_STARTED = "object_started"

    # Mask events
    MASK_LOADED = "mask_loaded"
    MASK_SAVED = "mask_saved"
    MASK_UPDATED = "mask_updated"

    # Session events
    SESSION_STARTED = "session_started"
    SESSION_COMPLETED = "session_completed"
    STATE_CHANGED = "state_changed"


@dataclass
class AnnotationEvent:
    """Event that occurs during annotation."""

    event_type: EventType
    data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


class EventEmitter:
    """
    Simple event emitter for pub/sub pattern.

    Allows components to subscribe to events without tight coupling.
    """

    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}

    def on(self, event_type: EventType, callback: Callable[[AnnotationEvent], None]):
        """Subscribe to an event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def off(self, event_type: EventType, callback: Callable[[AnnotationEvent], None]):
        """Unsubscribe from an event type."""
        if event_type in self._listeners:
            self._listeners[event_type].remove(callback)

    def emit(self, event: AnnotationEvent):
        """Emit an event to all subscribers."""
        if event.event_type in self._listeners:
            for callback in self._listeners[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    # Log but don't crash on listener errors
                    print(f"Error in event listener: {e}")

    def clear(self):
        """Clear all event listeners."""
        self._listeners.clear()
