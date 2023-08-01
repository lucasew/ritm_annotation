import logging
import os
from pathlib import Path

import torch
import cv2
import numpy as np

from ritm_annotation.utils.exp import init_experiment
from ritm_annotation.utils.misc import load_module
from ritm_annotation.data.base import ISDataset
from ritm_annotation.data.sample import DSample

logger = logging.getLogger(__name__)


class AnnotationDataset(ISDataset):
    def __init__(
        self,
        images_path: Path,
        masks_path: Path,
        split="train",
        dry_run=False
    ):
        self.images_path = images_path
        self.masks_path = masks_path
        self.classes = set()
        self.items = []
        if not dry_run:
            for item in masks_path.iterdir():
                image_file = images_path / item.name
                if not item.is_dir():
                    logger.warn(f"AnnotationDataset: found impurities: {item}")
                    continue
                if not (image_file.exists() and image_file.is_file()):
                    logger.warn(f"Found mask for {item.name} but not image")
                    continue
                for mask_file in item.iterdir():
                    class_name = mask_file.stem
                    self.classes.add(class_name)
                self.items.append(item.name)
        self.items.sort()
        self.classes = list(self.classes)
        self.classes.sort()
        classes = self.classes
        self.classes = {}
        for i, class_name in enumerate(classes):
            classes[class_name] = i  # O(1) instead of O(n)

        total_amount = len(self.items)
        train_amount = int(total_amount * 0.7)
        val_amount = total_amount - train_amount
        if split == 'train':
            self.items = self.items[:train_amount]
        elif split == 'val':
            self.items = self.items[-val_amount:]
        else:
            raise ValueError("split must be either train or val")

    def get_sample(self, index: int) -> DSample:
        item = self.items[index]
        image_path = self.images_path / item
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        objects = []
        ignored_regions = []
        masks = []
        for i, mask_path in enumerate((self.masks_path / item).iterdir()):
            class_id = self.classes[mask_path.stem]
            gt_mask = cv2.imread(str(mask_path), 0).astype('int32')
            instances_mask = np.zeros_like(gt_mask)
            instances_mask[gt_mask > 0] = 1
            instances_mask[gt_mask == 0] = 2
            masks.append(instances_mask)
            objects.append((class_id, 1))
            ignored_regions.append((class_id, 2))
        return DSample(
            image,
            np.stack(masks, axis=2),
            objects_ids=objects,
            ignore_ids=ignored_regions,
            sample_id=index,
        )




COMMAND_DESCRIPTION = "Run finetune trains using a dataset in the format the annotator generates"


def command(parser):
    parser.add_argument(
        "model_path", type=Path, help="Path to the model script."
    )
    parser.add_argument(
        "images_path", type=Path, help="Path to the dataset images."
    )
    parser.add_argument(
        "masks_path", type=Path, help="Path to the dataset masks."
    )
    parser.add_argument(
        "-o", '--output',
        dest="experiment_path", type=Path, default=Path('.'), help="Where to store experiment data"
    )

    parser.add_argument('-n', '--num-epochs', dest='num_epochs', type=int, help="Amount of epochs")

    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help="Here you can specify the name of the experiment. "  # noqa:E501
        "It will be added as a suffix to the experiment folder.",
    )  # noqa:E501

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Dataloader threads.",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        type=int,
        default=-1,
        help="You can override model batch size by specify positive number.",
    )  # noqa:E501

    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Number of GPUs. "
        'If you only specify "--gpus" argument, the ngpus value will be calculated automatically. '  # noqa:E501
        'You should use either this argument or "--gpus".',
    )  # noqa:E501

    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        required=False,
        help='Ids of used GPUs. You should use either this argument or "--ngpus".',  # noqa: E501
    )  # noqa:E501

    parser.add_argument(
        "--resume-exp",
        type=str,
        default=None,
        help="The prefix of the name of the experiment to be continued. "  # noqa:E501
        'If you use this field, you must specify the "--resume-prefix" argument.',  # noqa:E501
    )  # noqa:E501

    parser.add_argument(
        "--resume-prefix",
        type=str,
        default="latest",
        help="The prefix of the name of the checkpoint to be loaded.",
    )  # noqa:E501

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="The number of the starting epoch from which training will continue. "  # noqa:E501
        "(it is important for correct logging and learning rate)",
    )  # noqa:E501

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Model weights will be loaded from the specified path if you use this argument.",  # noqa: E501
    )  # noqa:E501

    parser.add_argument(
        "--temp-model-path",
        type=str,
        default="",
        help="Do not use this argument (for internal purposes).",
    )  # noqa:E501

    parser.add_argument("--local_rank", type=int, default=0)

    def handle(args):
        model_path = Path(args.model_path)
        if args.temp_model_path != "":
            logger.debug(
                "Falling back to temp_model_path because model_path wasn't specified"  # noqa:E501
            )
            model_path = Path(args.temp_model_path)
        if not model_path.is_absolute():
            model_path = (
                Path(__file__).parent.parent.parent / "models" / model_path
            )
        model_path = model_path.resolve()
        args.model_path = model_path
        logger.debug(f"Final model path: '{model_path}'")

        model_script = load_module(model_path)

        model_base_name = model_script.__dict__.get('MODEL_NAME')

        args.distributed = "WORLD_SIZE" in os.environ
        cfg = init_experiment(args, model_base_name)
        for k, v in os.environ.items():
            if k.startswith("RITM_"):
                cfgkey = k.replace("RITM_", "")
                logger.warning(
                    f"Changing configuration entry from environment variable: {cfgkey}={v}"  # noqa:E501
                )  # noqa: E501
                cfg[cfgkey] = v

        torch.backends.cudnn.benchmark = True
        torch.multiprocessing.set_sharing_strategy("file_system")
        logger.debug("Basic validations passed")
        model, model_cfg = model_script.init_model(cfg)
        trainer = model_script.get_trainer(model, cfg, model_cfg)
        # TODO: dataset
        trainer.trainset = AnnotationDataset(
            images_path=args.images_path,
            masks_path=args.masks_path,
            split='train'
        )
        trainer.valset = AnnotationDataset(
            images_path=args.images_path,
            masks_path=args.masks_path,
            split='val'
        )
        if args.num_epochs is None:
            args.num_epochs = model_cfg.default_num_epochs
        trainer.run(num_epochs=args.num_epochs)

    return handle
