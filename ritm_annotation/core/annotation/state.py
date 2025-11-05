"""
State management for annotation sessions.

Contains data classes representing the state of an annotation session.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Click:
    """Represents a single click in the annotation."""

    x: float
    y: float
    is_positive: bool
    object_id: int = 0

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "is_positive": self.is_positive,
            "object_id": self.object_id,
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"],
            is_positive=data["is_positive"],
            object_id=data.get("object_id", 0),
        )


@dataclass
class ObjectState:
    """State of a single annotated object."""

    object_id: int
    clicks: List[Click] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    probability_map: Optional[np.ndarray] = None
    is_finished: bool = False

    def add_click(self, click: Click):
        """Add a click to this object."""
        self.clicks.append(click)

    def to_dict(self):
        """Convert to dictionary (without numpy arrays)."""
        return {
            "object_id": self.object_id,
            "clicks": [c.to_dict() for c in self.clicks],
            "is_finished": self.is_finished,
            "num_clicks": len(self.clicks),
        }


@dataclass
class AnnotationState:
    """
    Complete state of an annotation session for a single image.

    This can be serialized/deserialized for saving progress.
    """

    image_path: Optional[str] = None
    image_shape: Optional[Tuple[int, int, int]] = None
    current_object_id: int = 0
    objects: List[ObjectState] = field(default_factory=list)
    result_mask: Optional[np.ndarray] = None

    def get_current_object(self) -> ObjectState:
        """Get or create the current object being annotated."""
        # Find object with current_object_id
        for obj in self.objects:
            if obj.object_id == self.current_object_id and not obj.is_finished:
                return obj

        # Create new object if not found
        new_obj = ObjectState(object_id=self.current_object_id)
        self.objects.append(new_obj)
        return new_obj

    def finish_current_object(self):
        """Mark current object as finished and move to next."""
        current = self.get_current_object()
        current.is_finished = True
        self.current_object_id += 1

    def get_all_clicks(self) -> List[Click]:
        """Get all clicks from all objects."""
        clicks = []
        for obj in self.objects:
            clicks.extend(obj.clicks)
        return clicks

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "image_path": self.image_path,
            "image_shape": self.image_shape,
            "current_object_id": self.current_object_id,
            "objects": [obj.to_dict() for obj in self.objects],
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        state = cls(
            image_path=data.get("image_path"),
            image_shape=data.get("image_shape"),
            current_object_id=data.get("current_object_id", 0),
        )
        # Note: objects would need full reconstruction with masks
        return state
