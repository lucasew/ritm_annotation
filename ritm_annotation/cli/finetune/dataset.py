import logging
import random
from pathlib import Path

import cv2
from albumentations.augmentations.geometric import longest_max_size

from ritm_annotation.data.base import ISDataset
from ritm_annotation.data.sample import DSample
from ritm_annotation.utils.exp_imports.default import *

logger = logging.getLogger(__name__)


def get_train_augmentator(model_cfg):
    crop_size = model_cfg.crop_size
    return Compose(
        [
            UniformRandomResize(scale_range=(0.75, 1.40)),
            HorizontalFlip(),
            PadIfNeeded(
                min_height=crop_size[0], min_width=crop_size[1], border_mode=0
            ),
            RandomCrop(*crop_size),
            RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(-0.15, 0.4),
                p=0.75,
            ),
            RGBShift(
                r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75
            ),
        ],
        p=1.0,
    )


def get_val_augmentator(model_cfg):
    crop_size = model_cfg.crop_size
    return Compose(
        [
            PadIfNeeded(
                min_height=crop_size[0], min_width=crop_size[1], border_mode=0
            ),
            RandomCrop(*crop_size),
        ],
        p=1.0,
    )


def get_points_sampler(model_cfg):
    return MultiPointSampler(
        model_cfg.num_max_points,
        prob_gamma=0.8,
        merge_objects_prob=0.15,
        max_num_merged_objects=2,
    )


class AnnotationDataset(ISDataset):
    def __init__(
        self,
        images_path: Path,
        masks_path: Path,
        split="train",
        dry_run=False,
        # the idea here is to resize the image to speed up data ingestion and training  # noqa:E501
        max_bigger_dimension=None,
        **kwargs,
    ):
        super(AnnotationDataset, self).__init__(**kwargs)
        self.images_path = images_path
        self.masks_path = masks_path
        self.max_bigger_dimension = max_bigger_dimension
        self.dataset_samples = []
        if not dry_run:
            for item in masks_path.iterdir():
                image_file = images_path / item.name
                if not item.is_dir():
                    logger.warn(
                        _(
                            "AnnotationDataset: found impurities: {item}"
                        ).format(item=item)
                    )
                    continue
                if not (image_file.exists() and image_file.is_file()):
                    logger.warn(
                        _("Found mask for {item_name} but not image").format(
                            item_name=item.name
                        )
                    )
                    continue
                has_mask = False
                for mask_file in item.iterdir():
                    has_mask = True
                if has_mask:
                    self.dataset_samples.append(item.name)
        self.dataset_samples.sort()

        total_amount = len(self.dataset_samples)
        train_amount = int(total_amount * 0.8)
        val_amount = total_amount - train_amount
        if split == "train":
            self.dataset_samples = self.dataset_samples[:train_amount]
        elif split == "val":
            self.dataset_samples = self.dataset_samples[-val_amount:]
        else:
            raise ValueError(_("split must be either train or val"))

    def get_sample(self, index: int) -> DSample:
        item = self.dataset_samples[index]
        image_path = self.images_path / item
        masks_path = self.masks_path / item

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.max_bigger_dimension is not None:
            image = longest_max_size(
                image, self.max_bigger_dimension, cv2.INTER_LINEAR
            )
        (h, w, *_rest) = image.shape

        mask_path = random.choice(list(masks_path.iterdir()))
        gt_mask = cv2.imread(str(mask_path), 0)

        if self.max_bigger_dimension is not None:
            gt_mask = longest_max_size(
                gt_mask, self.max_bigger_dimension, cv2.INTER_NEAREST_EXACT
            )
        gt_mask[gt_mask > 0] = 1
        gt_mask = gt_mask.astype("int32")
        logger.debug(
            _("Processed item {index}: '{item}' (shape: ({w}, {h})").format(
                index=index, item=item, w=w, h=h
            )
        )
        return DSample(image, gt_mask, objects_ids=[1], sample_id=index)
