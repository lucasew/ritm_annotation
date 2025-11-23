"""
Example Web API for interactive annotation.

This demonstrates how the modular architecture allows
creating a web interface using the same core logic.

Requirements:
    pip install fastapi uvicorn python-multipart

Usage:
    python examples/web_api_example.py

Then visit http://localhost:8000/docs for API documentation.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import cv2
from pathlib import Path
import base64

# Import core annotation components
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ritm_annotation.core.annotation import AnnotationSession
from ritm_annotation.inference.utils import load_is_model


class Click(BaseModel):
    """Click model for API."""

    x: float
    y: float
    is_positive: bool


class AnnotationRequest(BaseModel):
    """Request to add clicks."""

    session_id: str
    clicks: List[Click]


class AnnotationResponse(BaseModel):
    """Response with prediction."""

    session_id: str
    mask_base64: str  # Base64 encoded PNG
    visualization_base64: str  # Base64 encoded PNG
    num_clicks: int


# Global session storage (in production, use Redis or database)
sessions = {}

app = FastAPI(
    title="Interactive Annotation API",
    description="Web API for interactive image segmentation",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model, predictor

    # Load model (update path as needed)
    checkpoint_path = "path/to/checkpoint.pth"
    device = "cuda"

    try:
        model = load_is_model(checkpoint_path, device)
        print(f"Model loaded from {checkpoint_path}")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will run in demo mode")
        model = None


@app.post("/session/create")
async def create_session(
    image: UploadFile = File(...),
    prob_thresh: float = 0.5,
):
    """
    Create a new annotation session.

    Args:
        image: Image file to annotate
        prob_thresh: Probability threshold for mask generation

    Returns:
        Session ID
    """
    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create session
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    from ritm_annotation.inference.predictors import get_predictor

    predictor = get_predictor(
        model,
        device="cuda",
        prob_thresh=prob_thresh,
    )

    session = AnnotationSession(
        predictor=predictor,
        prob_thresh=prob_thresh,
    )
    session.load_image(img)

    # Generate session ID
    import uuid

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "session": session,
        "image": img,
    }

    return {
        "session_id": session_id,
        "image_shape": img.shape,
    }


@app.post("/session/{session_id}/click")
async def add_click(
    session_id: str,
    click: Click,
):
    """
    Add a click to the session.

    Args:
        session_id: Session identifier
        click: Click coordinates and type

    Returns:
        Updated prediction
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_id]
    session = session_data["session"]

    # Add click
    try:
        prob_map = session.add_click(
            x=click.x,
            y=click.y,
            is_positive=click.is_positive,
        )

        # Get visualization
        viz_data = session.get_visualization_data()

        # Encode mask as base64 PNG
        mask = (prob_map > 0.5).astype(np.uint8) * 255
        _, mask_buffer = cv2.imencode(".png", mask)
        mask_base64 = base64.b64encode(mask_buffer).decode("utf-8")

        # Create simple visualization
        vis = session_data["image"].copy()
        # Overlay mask
        if prob_map is not None:
            overlay = np.zeros_like(vis)
            overlay[:, :, 1] = (prob_map * 255).astype(np.uint8)
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        # Draw clicks
        for c in viz_data["clicks"]:
            color = (0, 255, 0) if c.is_positive else (255, 0, 0)
            cv2.circle(vis, (int(c.x), int(c.y)), 3, color, -1)

        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        _, vis_buffer = cv2.imencode(".png", vis_bgr)
        vis_base64 = base64.b64encode(vis_buffer).decode("utf-8")

        return {
            "session_id": session_id,
            "mask_base64": mask_base64,
            "visualization_base64": vis_base64,
            "num_clicks": len(viz_data["clicks"]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/undo")
async def undo_click(session_id: str):
    """Undo last click."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]["session"]
    success = session.undo_click()

    return {"success": success}


@app.post("/session/{session_id}/reset")
async def reset_clicks(session_id: str):
    """Reset all clicks for current object."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]["session"]
    session.reset_clicks()

    return {"success": True}


@app.post("/session/{session_id}/finish_object")
async def finish_object(session_id: str):
    """Finish current object and start new one."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]["session"]
    session.finish_object()

    viz_data = session.get_visualization_data()

    return {
        "success": True,
        "current_object_id": viz_data["current_object_id"],
        "num_finished_objects": viz_data["num_objects"],
    }


@app.get("/session/{session_id}/result")
async def get_result_mask(session_id: str):
    """Get final result mask."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]["session"]
    result_mask = session.get_result_mask()

    if result_mask is None:
        return {"mask_base64": None}

    # Encode as PNG
    _, buffer = cv2.imencode(".png", result_mask.astype(np.uint8))
    mask_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "mask_base64": mask_base64,
        "num_objects": len(np.unique(result_mask)) - 1,
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete annotation session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    return {"success": True}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "active_sessions": len(sessions),
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting Interactive Annotation API...")
    print("Visit http://localhost:8000/docs for API documentation")

    uvicorn.run(app, host="0.0.0.0", port=8000)
