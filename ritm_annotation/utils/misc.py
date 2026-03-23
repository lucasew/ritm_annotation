import hashlib
import importlib
import itertools
import logging
import urllib.request
from gettext import gettext as _
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def get_dims_with_exclusion(dim, exclude=None):
    """
    Generates a list of dimensions up to `dim`, optionally omitting a specific dimension.

    Useful for creating dimension lists for tensor operations (e.g., reductions like `mean()`
    or `sum()`) where you want to aggregate across all axes except the batch or channel dimension.

    Args:
        dim: Total number of dimensions to generate (0 to dim-1).
        exclude: The specific dimension index to omit from the returned list.

    Returns:
        List of dimension indices.
    """
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def save_checkpoint(net, checkpoints_path, epoch=None, prefix="", multi_gpu=False):
    """
    Serializes and saves a model's state dictionary and configuration to disk.

    If an epoch is provided, it generates an ordered filename (e.g., `001.pth`), otherwise
    it overwrites the `last_checkpoint.pth`. This is critical for resuming training or
    exporting models for inference.

    Side Effects:
        - Creates the `checkpoints_path` directory if it does not exist.
        - Writes a .pth file containing the `state_dict` and `_config`.
        - Unwraps `DataParallel` or `DistributedDataParallel` modules if `multi_gpu` is True.

    Args:
        net: The neural network model to save. Must have `state_dict()` and `_config`.
        checkpoints_path: The directory path (Path object) where the checkpoint will be saved.
        epoch: The current training epoch. Used for naming the file.
        prefix: Optional prefix to prepend to the checkpoint filename (e.g., 'best').
        multi_gpu: Set to True if the model is wrapped in DDP or DP to extract the underlying module.
    """
    if epoch is None:
        checkpoint_name = "last_checkpoint.pth"
    else:
        checkpoint_name = f"{epoch:03d}.pth"

    if prefix:
        checkpoint_name = f"{prefix}_{checkpoint_name}"

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    logger.debug(
        _("Save checkpoint to {checkpoint_path}").format(
            checkpoint_path=str(checkpoint_path)
        )
    )

    net = net.module if multi_gpu else net
    torch.save(
        {"state_dict": net.state_dict(), "config": net._config},
        str(checkpoint_path),
    )


def get_bbox_from_mask(mask):
    """
    Computes the tightest bounding box encompassing all non-zero elements in a 2D mask.

    This avoids iterating over individual pixels by aggressively reducing rows and columns
    via `np.any`, making it highly efficient for generating ground-truth crop coordinates
    from segmentation masks.

    Args:
        mask: A 2D numpy array representing a segmentation mask.

    Returns:
        Tuple of (rmin, rmax, cmin, cmax) denoting the bounding box coordinates.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expand_ratio, min_crop_size=None):
    """
    Scales a bounding box uniformly around its center point.

    Essential for adding spatial context around tight object crops during training or
    inference. The expanded box maintains the original center coordinates.

    Args:
        bbox: Tuple of (rmin, rmax, cmin, cmax) representing the original bounding box.
        expand_ratio: Multiplier for the height and width (e.g., 1.2 adds 20% context).
        min_crop_size: Ensures the resulting width and height are at least this large.

    Returns:
        Tuple of the expanded bounding box coordinates (rmin, rmax, cmin, cmax).
        Note: The returned coordinates might fall outside the image boundaries and
        require clamping via `clamp_bbox`.
    """
    rmin, rmax, cmin, cmax = bbox
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expand_ratio * (rmax - rmin + 1)
    width = expand_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax


def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
    """
    Restricts bounding box coordinates to lie within a specified spatial domain.

    Usually applied after `expand_bbox` or random augmentation to prevent out-of-bounds
    indexing errors during image cropping.

    Args:
        bbox: The bounding box to clamp (rmin, rmax, cmin, cmax).
        rmin, rmax, cmin, cmax: The boundary constraints (typically 0 and image dimensions).

    Returns:
        A clamped tuple (rmin, rmax, cmin, cmax).
    """
    return (
        max(rmin, bbox[0]),
        min(rmax, bbox[1]),
        max(cmin, bbox[2]),
        min(cmax, bbox[3]),
    )


def get_bbox_iou(b1, b2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    This implementation factors the 2D bounding box IoU into the product of
    1D segment IoUs along the height and width axes.

    Args:
        b1, b2: Bounding boxes defined as tuples (rmin, rmax, cmin, cmax).
    """
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
    """
    Computes the 1D Intersection over Union for two line segments.

    Edge case: Adds 1e-6 to the union to prevent division by zero in degenerate cases.

    Args:
        s1, s2: 1D segments defined as tuples (min_val, max_val).
    """
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union


def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()


def ignore_params_then_call(func):
    def ret(*args, **kwargs):
        return func()

    return ret


def load_module(script_path, module_name="module"):
    logger.debug(_("Loading module '{module_path}'...").format(module_path=script_path))
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None, _("Can't import module at '{module_path}'").format(
        module_path=script_path
    )
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


def get_default_weight():
    """
    Ensures the default HRNet18 interactive segmentation weights are locally available.

    This function acts as an auto-downloader for the primary model checkpoint required for
    out-of-the-box inference. It checks the local user cache directory; if the file is
    missing, it streams the download with a progress bar and verifies its SHA256 integrity.

    Side Effects:
        - Creates `~/.cache/ritm_annotation` if missing.
        - Streams network requests to download the `.pth` file.
        - Unlinks (deletes) the partially downloaded file if an exception occurs during transfer.

    Returns:
        Pathlib object pointing to the cached `.pth` weight file.
    """
    OUTPUT_DIR = Path.home() / ".cache" / "ritm_annotation"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_MODEL_URL = "https://github.com/SamsungLabs/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h18_itermask.pth"
    DEFAULT_MODEL_FILE = OUTPUT_DIR / "coco_lvis_h18_itermask.pth"
    DEFAULT_MODEL_SHA256 = (
        "5f69cfce354d1507e3850bfc39ee7057c8dd27b6a4910d1d2dc724916b9ee32b"
    )
    if DEFAULT_MODEL_FILE.exists():
        return DEFAULT_MODEL_FILE
    try:
        hasher = hashlib.sha256()
        with urllib.request.urlopen(DEFAULT_MODEL_URL) as req:
            with DEFAULT_MODEL_FILE.open("wb") as f:
                file_size = int(req.headers["Content-Length"])
                ops = tqdm(
                    total=file_size,
                    desc=_("Downloading") + f" {DEFAULT_MODEL_URL}",
                )
                while True:
                    buf = req.read(16 * 1024)
                    if not buf:
                        break
                    hasher.update(buf)
                    f.write(buf)
                    ops.update(len(buf))
        if hasher.hexdigest() != DEFAULT_MODEL_SHA256:
            logger.warning(
                _(
                    "SHA256 of default model is {actual_hash}, expected {expected_hash}"
                ).format(
                    actual_hash=hasher.hexdigest(),
                    expected_hash=DEFAULT_MODEL_SHA256,
                )
            )
        return DEFAULT_MODEL_FILE
    except Exception as e:  # se erro deletar o arquivo
        if DEFAULT_MODEL_FILE.exists():
            DEFAULT_MODEL_FILE.unlink()
        import traceback

        traceback.print_exc()
        raise e


def try_tqdm(items, desc="", **kwargs):
    """
    Wraps an iterable with a progress bar.

    Provides a centralized fallback mechanism. It uses `tqdm.auto` to automatically
    determine if the environment is a Jupyter notebook or a standard terminal,
    preventing messy line breaks in console outputs.
    """
    return tqdm(items, desc=desc, **kwargs)


def incrf():
    """
    Returns an infinite iterator starting from 1.

    Useful for generating sequential IDs (e.g., generating unique object instance IDs)
    without managing external counter state.
    """
    return itertools.count(1)
