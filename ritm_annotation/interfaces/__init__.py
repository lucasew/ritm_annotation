"""
Interfaces module - UI adapters for annotation core.

Provides adapters to connect the core annotation logic
with different UI frameworks (Tkinter, Web, etc).
"""

from .gui_adapter import GUIAnnotationAdapter

__all__ = ['GUIAnnotationAdapter']
