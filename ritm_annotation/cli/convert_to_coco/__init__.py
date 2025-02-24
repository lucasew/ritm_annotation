import logging
import time
from gettext import gettext as _
from pathlib import Path

from ritm_annotation.utils.misc import incrf, try_tqdm

logger = logging.getLogger(__name__)

COMMAND_DESCRIPTION = _("Convert dataset in mask form to COCO")


def command(subparser):
    subparser.add_argument("input", type=Path, help=_("Dataset folder in mask form"))
    subparser.add_argument(
        "output", type=Path, help=_("Where to save the COCO dataset JSON")
    )
    subparser.add_argument(
        "--overwrite",
        action="store_true",
        help=_("Overwrite JSON file if it exists"),
    )
    subparser.add_argument(
        "--description",
        type=str,
        help=_("Description for the COCO dataset"),
        default=_("Created with ritm_annotation"),
    )

    def handle(args):
        import cv2
        import numpy as np
        import pycocotools.mask

        images_idx = incrf()
        annotations_idx = incrf()
        data = dict(
            meta=dict(
                contributor="Created with ritm_annotation",
                description=args.description,
                date_created=time.strftime("%Y/%m/%d"),
                version="1.0",
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                year=time.strftime("%Y"),
            ),
            license=[
                dict(
                    id=1,
                    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    name="ritm (edit later)",
                )
            ],
            images=[],
            annotations=[],
            categories=[],
        )
        assert args.input.exists() and args.input.is_dir(), _(
            "Dataset folder must exist and be a folder"
        )
        if not args.overwrite:
            assert not args.output.exists(), _(
                "COCO dataset exists, use --overwrite to ignore this"
            )
        input_items = list(args.input.iterdir())
        args.output.parent.mkdir(exist_ok=True, parents=True)

        classes = set()
        for item in try_tqdm(input_items, desc=_("Enumerating all classes...")):
            if not item.is_dir():
                continue
            for subitem in item.iterdir():
                if subitem.name.endswith(".json"):
                    continue
                classes.add(subitem.stem)  # only the part without the extension
        classes = list(classes)
        classes.sort()
        classes = {cls: i + 1 for i, cls in enumerate(classes)}
        for cls, i in classes.items():
            data["categories"].append(dict(id=i, name=cls, supercategory=None))
        for item in try_tqdm(input_items, desc=_("Ingesting images...")):
            if not item.is_dir():
                continue
            image_id = next(images_idx)
            image_to_append = dict(license=1, file_name=item.name, id=image_id)
            have_annotation = False
            for subitem in item.iterdir():
                if subitem.name.endswith(".json"):
                    continue
                segm = cv2.imread(str(subitem), 0)
                if segm is None:
                    continue
                if np.count_nonzero(segm) == 0:
                    continue
                annotation_id = next(annotations_idx)
                # segm = segm != 0
                pos = np.where(segm != 0)
                try:
                    np.min(pos[1])
                except ValueError:  # empty mask
                    continue
                (h, w) = segm.shape
                image_to_append["width"] = w
                image_to_append["height"] = h
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmin - xmax) < 1 or abs(ymin - ymax) < 1:
                    continue  # pular máscaras vazias
                class_mask_fortran = np.asarray(segm, dtype="uint8", order="F")
                rle = pycocotools.mask.encode(class_mask_fortran)
                rle["counts"] = rle["counts"].decode("utf-8")
                data["annotations"].append(
                    dict(
                        segmentation=rle,
                        area=int(pycocotools.mask.area(rle)),
                        iscrowd=0,
                        image_id=image_id,
                        bbox=list(pycocotools.mask.toBbox(rle)),
                        category_id=classes[subitem.stem],
                        id=annotation_id,
                    )
                )
                have_annotation = True
            if have_annotation:
                data["images"].append(image_to_append)
        with args.output.open("w") as f:
            from json import dump

            dump(data, f)

    return handle
