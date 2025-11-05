"""
Pure utility functions for annotation logic.

These functions have no side effects and can be tested in isolation.
"""

import numpy as np
from typing import Tuple, List, Optional
import cv2


def apply_mask_threshold(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert probability map to binary mask.

    Args:
        prob_map: Float array [0, 1]
        threshold: Threshold value

    Returns:
        Binary mask (0 or 1)
    """
    return (prob_map > threshold).astype(np.uint8)


def merge_mask_with_result(
    result_mask: np.ndarray, new_mask: np.ndarray, object_id: int
) -> np.ndarray:
    """
    Merge new object mask into result mask.

    Args:
        result_mask: Existing result mask with object IDs
        new_mask: Binary mask for new object
        object_id: ID to assign to new object

    Returns:
        Updated result mask
    """
    result = result_mask.copy()
    object_pixels = new_mask > 0
    result[object_pixels] = object_id + 1  # +1 because 0 is background
    return result


def create_colored_mask(mask: np.ndarray, colormap: str = "tab20") -> np.ndarray:
    """
    Create colored visualization of segmentation mask.

    Args:
        mask: Integer mask with object IDs
        colormap: Matplotlib colormap name

    Returns:
        RGB colored mask
    """
    import matplotlib.pyplot as plt

    if mask is None or np.max(mask) == 0:
        return np.zeros((*mask.shape, 3), dtype=np.uint8)

    cmap = plt.get_cmap(colormap)
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]

    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, obj_id in enumerate(unique_ids):
        color = np.array(cmap(idx / max(len(unique_ids), 1))[:3]) * 255
        obj_mask = mask == obj_id
        colored[obj_mask] = color.astype(np.uint8)

    return colored


def blend_image_with_mask(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, colormap: str = "tab20"
) -> np.ndarray:
    """
    Blend image with colored segmentation mask.

    Args:
        image: RGB image
        mask: Integer mask with object IDs
        alpha: Blending factor [0, 1]
        colormap: Matplotlib colormap

    Returns:
        Blended image
    """
    if mask is None or np.max(mask) == 0:
        return image.copy()

    colored_mask = create_colored_mask(mask, colormap)
    mask_area = mask > 0

    result = image.copy()
    result[mask_area] = (
        alpha * colored_mask[mask_area] + (1 - alpha) * image[mask_area]
    ).astype(np.uint8)

    return result


def draw_clicks_on_image(
    image: np.ndarray,
    clicks: List,
    click_radius: int = 3,
    positive_color: Tuple[int, int, int] = (0, 255, 0),
    negative_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Draw click indicators on image.

    Args:
        image: RGB image
        clicks: List of Click objects
        click_radius: Radius for click circles
        positive_color: Color for positive clicks (RGB)
        negative_color: Color for negative clicks (RGB)

    Returns:
        Image with clicks drawn
    """
    result = image.copy()

    for click in clicks:
        color = positive_color if click.is_positive else negative_color
        center = (int(click.x), int(click.y))

        # Draw filled circle
        cv2.circle(result, center, click_radius, color, -1)

        # Draw white border
        cv2.circle(result, center, click_radius + 1, (255, 255, 255), 1)

    return result


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two masks.

    Args:
        mask1: Binary mask
        mask2: Binary mask

    Returns:
        IoU score [0, 1]
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()

    if union == 0:
        return 0.0

    return intersection / union


def validate_image(image: np.ndarray) -> None:
    """
    Validate that image has correct format.

    Raises:
        ValueError: If image is invalid
    """
    if image is None:
        raise ValueError("Image is None")

    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image must be numpy array, got {type(image)}")

    if image.ndim != 3:
        raise ValueError(f"Image must be 3D (H, W, C), got shape {image.shape}")

    if image.shape[2] != 3:
        raise ValueError(f"Image must have 3 channels, got {image.shape[2]}")

    if image.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValueError(f"Invalid image dtype: {image.dtype}")


def validate_mask(mask: np.ndarray, image_shape: Tuple[int, int]) -> None:
    """
    Validate that mask has correct format.

    Args:
        mask: Segmentation mask
        image_shape: Expected (height, width)

    Raises:
        ValueError: If mask is invalid
    """
    if mask is None:
        raise ValueError("Mask is None")

    if not isinstance(mask, np.ndarray):
        raise ValueError(f"Mask must be numpy array, got {type(mask)}")

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D (H, W), got shape {mask.shape}")

    if mask.shape != image_shape:
        raise ValueError(
            f"Mask shape {mask.shape} doesn't match image shape {image_shape}"
        )


def compute_click_statistics(clicks: List) -> dict:
    """
    Compute statistics about clicks.

    Args:
        clicks: List of Click objects

    Returns:
        Dictionary with statistics
    """
    if not clicks:
        return {
            "num_total": 0,
            "num_positive": 0,
            "num_negative": 0,
            "ratio_positive": 0.0,
        }

    num_positive = sum(1 for c in clicks if c.is_positive)
    num_negative = len(clicks) - num_positive

    return {
        "num_total": len(clicks),
        "num_positive": num_positive,
        "num_negative": num_negative,
        "ratio_positive": num_positive / len(clicks) if clicks else 0.0,
    }


def estimate_object_center(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Estimate center of object in mask using centroid.

    Args:
        mask: Binary mask

    Returns:
        (x, y) coordinates of center, or None if mask is empty
    """
    if mask is None or mask.sum() == 0:
        return None

    # Compute centroid
    y_coords, x_coords = np.where(mask > 0)

    if len(y_coords) == 0:
        return None

    cx = int(np.mean(x_coords))
    cy = int(np.mean(y_coords))

    return (cx, cy)


def find_boundary_points(
    mask: np.ndarray, num_points: int = 8
) -> List[Tuple[int, int]]:
    """
    Find points on object boundary.

    Args:
        mask: Binary mask
        num_points: Number of boundary points to return

    Returns:
        List of (x, y) coordinates on boundary
    """
    if mask is None or mask.sum() == 0:
        return []

    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < num_points:
        # Return all points if contour is small
        return [(int(pt[0][0]), int(pt[0][1])) for pt in largest_contour]

    # Sample evenly spaced points
    indices = np.linspace(0, len(largest_contour) - 1, num_points, dtype=int)
    points = [
        (int(largest_contour[i][0][0]), int(largest_contour[i][0][1])) for i in indices
    ]

    return points
