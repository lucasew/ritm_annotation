import os
import pickle as pkl
import random
from gettext import gettext as _
from pathlib import Path

import cv2
import numpy as np

from ritm_annotation.data.base import ISDataset
from ritm_annotation.data.sample import DSample
from ritm_annotation.utils.misc import get_labels_with_sizes


class ADE20kDataset(ISDataset):
    def __init__(
        self,
        dataset_path,
        split="train",
        stuff_prob=0.0,
        dry_run=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert split in {"train", "val"}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self.dataset_split_folder = "training" if split == "train" else "validation"
        self.stuff_prob = stuff_prob

        anno_path = self.dataset_path / f"{split}-annotations-object-segmentation.pkl"
        if os.path.exists(anno_path):
            with anno_path.open("rb") as f:
                annotations = pkl.load(f)
        elif dry_run:
            annotations = {}
        else:
            raise RuntimeError(
                _("Can't find annotations at {anno_path}").format(anno_path=anno_path)
            )
        self.annotations = annotations
        self.dataset_samples = list(annotations.keys())

    def get_sample(self, index) -> DSample:
        image_id = self.dataset_samples[index]
        sample_annos = self.annotations[image_id]

        image_path = str(self.dataset_path / sample_annos["folder"] / f"{image_id}.jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # select random mask for an image
        layer = random.choice(sample_annos["layers"])
        mask_path = str(self.dataset_path / sample_annos["folder"] / layer["mask_name"])
        instances_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[
            :, :, 0
        ]  # the B channel holds instances
        instances_mask = instances_mask.astype(np.int32)
        object_ids, _ = get_labels_with_sizes(instances_mask)

        if (self.stuff_prob <= 0) or (random.random() > self.stuff_prob):
            # remove stuff objects
            for i, object_id in enumerate(object_ids):
                if i in layer["stuff_instances"]:
                    instances_mask[instances_mask == object_id] = 0
            object_ids, _ = get_labels_with_sizes(instances_mask)

        return DSample(image, instances_mask, objects_ids=object_ids, sample_id=index)
