import cv2
import numpy as np
from pathlib import Path

from ritm_annotation.data.base import ISDataset
from ritm_annotation.data.sample import DSample
from ritm_annotation.utils.exp_imports.default import *


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


class AnnotationDataset(ISDataset):
    def __init__(
        self,
        images_path: Path,
        masks_path: Path,
        split="train",
        dry_run=False,
        **kwargs
    ):
        super(AnnotationDataset, self).__init__(**kwargs)
        self.images_path = images_path
        self.masks_path = masks_path
        # self.classes = set()
        self.dataset_samples = []
        if not dry_run:
            for item in masks_path.iterdir():
                image_file = images_path / item.name
                if not item.is_dir():
                    logger.warn(f"AnnotationDataset: found impurities: {item}")
                    continue
                if not (image_file.exists() and image_file.is_file()):
                    logger.warn(f"Found mask for {item.name} but not image")
                    continue
                # for mask_file in item.iterdir():
                #     class_name = mask_file.stem
                #     self.classes.add(class_name)
                self.dataset_samples.append(item.name)
        self.dataset_samples.sort()
        # self.classes = list(self.classes)
        # self.classes.sort()
        # classes = self.classes
        # self.classes = {}
        # for i, class_name in enumerate(classes):
        #     self.classes[class_name] = i  # O(1) instead of O(n)

        total_amount = len(self.dataset_samples)
        train_amount = int(total_amount * 0.8)
        val_amount = total_amount - train_amount
        if split == 'train':
            self.dataset_samples = self.dataset_samples[:train_amount]
        elif split == 'val':
            self.dataset_samples = self.dataset_samples[-val_amount:]
        else:
            raise ValueError("split must be either train or val")

    def get_sample(self, index: int) -> DSample:
        item = self.dataset_samples[index]
        image_path = self.images_path / item
        masks_path = self.masks_path / item

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        objects = []
        ignored_regions = []
        (w, h, *_rest) = image.shape
        # masks = np.zeros((len(self.classes), w, h), dtype='int32')
        masks = []
        for i, mask_path in enumerate(masks_path.iterdir()):
            # class_id = self.classes[mask_path.stem]
            gt_mask = cv2.imread(str(mask_path), 0).astype('int32')
            instances_mask = np.zeros_like(gt_mask)
            instances_mask[gt_mask > 0] = 1
            instances_mask[gt_mask == 0] = 2
            # masks[class_id, :, :] = instances_mask
            masks.append(instances_mask)
            # objects.append((class_id, 1))
            objects.append((i, 1))
            # ignored_regions.append((class_id, 2))
            ignored_regions.append((i, 2))
        logger.debug(f"Processed item {index}: '{item}'")
        return DSample(
            image,
            np.stack(masks, axis=2),
            objects_ids=objects,
            ignore_ids=ignored_regions,
            sample_id=index,
        )


