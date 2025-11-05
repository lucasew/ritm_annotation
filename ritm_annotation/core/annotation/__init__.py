"""
Core annotation module - UI-agnostic annotation logic.

This module provides the base abstractions for interactive annotation
that can be used with any UI framework (Tkinter, Web, CLI, etc).
"""

from .session import AnnotationSession
from .events import AnnotationEvent, EventType, EventEmitter
from .state import AnnotationState, ObjectState, Click

__all__ = [
    "AnnotationSession",
    "AnnotationEvent",
    "EventType",
    "EventEmitter",
    "AnnotationState",
    "ObjectState",
    "Click",
]
