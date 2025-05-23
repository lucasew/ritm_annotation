import os
import pickle as pkl
import random
from gettext import gettext as _
from pathlib import Path

import cv2
import numpy as np

from ritm_annotation.data.base import ISDataset
from ritm_annotation.data.sample import DSample


class OpenImagesDataset(ISDataset):
    def __init__(self, dataset_path, split="train", dry_run=False, **kwargs):
        super().__init__(**kwargs)
        assert split in {"train", "val", "test"}

        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path = self._split_path / "images"
        self._masks_path = self._split_path / "masks"
        self.dataset_split = split

        clean_anno_path = (
            self._split_path / f"{split}-annotations-object-segmentation_clean.pkl"
        )
        if os.path.exists(clean_anno_path):
            with clean_anno_path.open("rb") as f:
                annotations = pkl.load(f)
        elif dry_run:
            annotations = dict(image_id_to_masks=[], dataset_samples=[])
        else:
            raise RuntimeError(
                _("Can't find annotations at {anno_path}").format(
                    anno_path=clean_anno_path
                )
            )
        self.image_id_to_masks = annotations["image_id_to_masks"]
        self.dataset_samples = annotations["dataset_samples"]

    def get_sample(self, index) -> DSample:
        image_id = self.dataset_samples[index]

        image_path = str(self._images_path / f"{image_id}.jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_paths = self.image_id_to_masks[image_id]
        # select random mask for an image
        mask_path = str(self._masks_path / random.choice(mask_paths))
        instances_mask = cv2.imread(mask_path)
        instances_mask = cv2.cvtColor(instances_mask, cv2.COLOR_BGR2GRAY)
        instances_mask[instances_mask > 0] = 1
        instances_mask = instances_mask.astype(np.int32)

        min_width = min(image.shape[1], instances_mask.shape[1])
        min_height = min(image.shape[0], instances_mask.shape[0])

        if image.shape[0] != min_height or image.shape[1] != min_width:
            image = cv2.resize(
                image, (min_width, min_height), interpolation=cv2.INTER_LINEAR
            )
        if (
            instances_mask.shape[0] != min_height
            or instances_mask.shape[1] != min_width
        ):
            instances_mask = cv2.resize(
                instances_mask,
                (min_width, min_height),
                interpolation=cv2.INTER_NEAREST,
            )

        object_ids = [1] if instances_mask.sum() > 0 else []

        return DSample(image, instances_mask, objects_ids=object_ids, sample_id=index)
