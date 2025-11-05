"""
GUI adapter for annotation session.

Bridges the AnnotationSession with Tkinter GUI components.
"""

from typing import Optional, Callable
import numpy as np
import cv2

from ..core.annotation import AnnotationSession, AnnotationEvent, EventType


class GUIAnnotationAdapter:
    """
    Adapter connecting AnnotationSession to Tkinter GUI.

    Provides a compatibility layer that:
    - Wraps AnnotationSession with GUI-friendly methods
    - Translates events to GUI callbacks
    - Handles visualization rendering
    """

    def __init__(
        self,
        session: AnnotationSession,
        update_image_callback: Optional[Callable] = None,
        click_radius: int = 3,
    ):
        """
        Initialize adapter.

        Args:
            session: Core annotation session
            update_image_callback: Callback to update GUI image
            click_radius: Radius for drawing click indicators
        """
        self.session = session
        self.update_image_callback = update_image_callback
        self.click_radius = click_radius

        # Subscribe to session events
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup event handlers for session events."""
        self.session.events.on(
            EventType.PREDICTION_COMPLETED,
            self._on_prediction_completed
        )
        self.session.events.on(
            EventType.CLICK_ADDED,
            self._on_click_added
        )
        self.session.events.on(
            EventType.OBJECT_FINISHED,
            self._on_object_finished
        )

    def _on_prediction_completed(self, event: AnnotationEvent):
        """Handle prediction completion."""
        if self.update_image_callback:
            self.update_image_callback()

    def _on_click_added(self, event: AnnotationEvent):
        """Handle click addition."""
        if self.update_image_callback:
            self.update_image_callback()

    def _on_object_finished(self, event: AnnotationEvent):
        """Handle object completion."""
        if self.update_image_callback:
            self.update_image_callback()

    # Compatibility methods for existing GUI code

    def set_image(self, image: np.ndarray):
        """Load image (compatible with old controller)."""
        self.session.load_image(image)

    def set_mask(self, mask: np.ndarray):
        """Load mask (compatible with old controller)."""
        self.session.load_mask(mask)

    def add_click(self, x: float, y: float, is_positive: bool):
        """Add click (compatible with old controller)."""
        self.session.add_click(x, y, is_positive)

    def undo_click(self):
        """Undo click (compatible with old controller)."""
        return self.session.undo_click()

    def partially_finish_object(self):
        """Finish object (compatible with old controller)."""
        self.session.finish_object()

    def finish_object(self):
        """Finish object and get result mask."""
        self.session.finish_object()
        return self.session.get_result_mask()

    def reset_clicks(self):
        """Reset clicks for current object."""
        self.session.reset_clicks()

    def get_visualization(
        self,
        alpha_blend: float = 0.5,
        click_radius: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get visualization for display.

        Args:
            alpha_blend: Blending factor for overlay
            click_radius: Radius for click indicators

        Returns:
            RGB visualization image
        """
        if click_radius is None:
            click_radius = self.click_radius

        # Get visualization data from session
        viz_data = self.session.get_visualization_data()

        image = viz_data['image']
        if image is None:
            return None

        # Create base visualization
        vis = image.copy()

        # Overlay result mask (finished objects)
        if viz_data['result_mask'] is not None:
            vis = self._overlay_mask(
                vis,
                viz_data['result_mask'],
                alpha=alpha_blend,
                colormap='tab20'
            )

        # Overlay current prediction
        if viz_data['current_prob_map'] is not None:
            # Convert probability map to binary mask
            current_mask = (viz_data['current_prob_map'] > 0.5).astype(np.uint8)
            # Use a different color for current object
            current_id = viz_data['current_object_id'] + 100
            vis = self._overlay_mask(
                vis,
                current_mask * current_id,
                alpha=alpha_blend * 0.7,
                colormap='viridis'
            )

        # Draw clicks
        for click in viz_data['clicks']:
            color = (0, 255, 0) if click.is_positive else (255, 0, 0)
            cv2.circle(
                vis,
                (int(click.x), int(click.y)),
                click_radius,
                color,
                -1
            )
            # Draw border
            cv2.circle(
                vis,
                (int(click.x), int(click.y)),
                click_radius + 1,
                (255, 255, 255),
                1
            )

        return vis

    def _overlay_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        colormap: str = 'tab20'
    ) -> np.ndarray:
        """
        Overlay segmentation mask on image.

        Args:
            image: RGB image
            mask: Integer mask with object IDs
            alpha: Blending factor
            colormap: Matplotlib colormap name

        Returns:
            Blended image
        """
        if mask is None or np.max(mask) == 0:
            return image

        # Create colored mask
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)

        # Normalize mask values
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]

        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for idx, obj_id in enumerate(unique_ids):
            color = np.array(cmap(idx / len(unique_ids))[:3]) * 255
            obj_mask = mask == obj_id
            colored_mask[obj_mask] = color.astype(np.uint8)

        # Blend with original image
        mask_area = mask > 0
        result = image.copy()
        result[mask_area] = (
            alpha * colored_mask[mask_area] +
            (1 - alpha) * image[mask_area]
        ).astype(np.uint8)

        return result

    @property
    def result_mask(self):
        """Get result mask (property for compatibility)."""
        return self.session.get_result_mask()

    @property
    def current_object_prob(self):
        """Get current object probability map."""
        return self.session.get_current_prediction()

    @property
    def probs_history(self):
        """Get probability history (for compatibility)."""
        # Return empty list - new session doesn't track full history
        return []

    def get_states(self):
        """Get states (for compatibility with old controller)."""
        # Return empty - undo is handled internally
        return None

    def set_states(self, states):
        """Set states (for compatibility)."""
        pass
