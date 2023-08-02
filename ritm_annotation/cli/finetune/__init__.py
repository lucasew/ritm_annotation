import logging
import os
from pathlib import Path

import torch

from ritm_annotation.utils.exp import init_experiment
from ritm_annotation.utils.misc import load_module

from .dataset import (
    AnnotationDataset,
    get_points_sampler,
    get_train_augmentator,
    get_val_augmentator,
)

logger = logging.getLogger(__name__)


COMMAND_DESCRIPTION = (
    "Run finetune trains using a dataset in the format the annotator generates"
)


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
        "-o",
        "--output",
        dest="experiment_path",
        type=Path,
        default=Path("."),
        help="Where to store experiment data",
    )

    parser.add_argument(
        "-n",
        "--num-epochs",
        dest="num_epochs",
        type=int,
        help="Amount of epochs",
    )

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

        model_base_name = model_script.__dict__.get("MODEL_NAME")

        args.distributed = "WORLD_SIZE" in os.environ
        cfg = init_experiment(args, model_base_name)
        cfg.weights = args.weights
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

        train_augmentator = get_train_augmentator(model_cfg)
        val_augmentator = get_val_augmentator(model_cfg)

        trainer = model_script.get_trainer(
            model, cfg, model_cfg, no_dataset=True
        )
        points_sampler = get_points_sampler(model_cfg)
        trainer.trainset = AnnotationDataset(
            images_path=args.images_path,
            masks_path=args.masks_path,
            split="train",
            augmentator=train_augmentator,
            max_bigger_dimension=1024,
            keep_background_prob=0.05,
            min_object_area=1000,
            points_sampler=points_sampler,
        )
        trainer.valset = AnnotationDataset(
            images_path=args.images_path,
            masks_path=args.masks_path,
            split="val",
            augmentator=val_augmentator,
            max_bigger_dimension=1024,
            min_object_area=1000,
            keep_background_prob=0.05,
            points_sampler=points_sampler,
        )
        if args.num_epochs is None:
            args.num_epochs = model_cfg.default_num_epochs
        trainer._before_needed_hook()
        assert trainer.lr_scheduler is not None
        trainer.run(num_epochs=args.num_epochs)

    return handle
