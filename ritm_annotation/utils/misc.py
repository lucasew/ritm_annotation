import importlib
import logging
from gettext import gettext as _
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def save_checkpoint(net, checkpoints_path, epoch=None, prefix="", multi_gpu=False):
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
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expand_ratio, min_crop_size=None):
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
    return (
        max(rmin, bbox[0]),
        min(rmax, bbox[1]),
        max(cmin, bbox[2]),
        min(cmax, bbox[3]),
    )


def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
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
    from hashlib import sha256
    from urllib.request import urlopen

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
        hasher = sha256()
        with urlopen(DEFAULT_MODEL_URL) as req:
            with DEFAULT_MODEL_FILE.open("wb") as f:
                file_size = int(req.headers["Content-Length"])
                print(file_size, type(file_size))
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
        DEFAULT_MODEL_FILE.unlink()
        import traceback

        traceback.print_exc()
        raise e


def try_tqdm(items, desc=""):
    try:
        if locals().get("get_ipython"):
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm
        return tqdm(list(items), desc=desc)
    except ImportError:
        logger.info(desc)
        return items


def incrf():
    i = 1
    while True:
        yield i
        i += 1
