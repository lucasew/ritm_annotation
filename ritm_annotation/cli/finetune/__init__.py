import logging
import os
from pathlib import Path

import torch

from ritm_annotation.utils.env import load_cfg_from_env
from ritm_annotation.utils.exp import init_experiment
from ritm_annotation.utils.misc import load_module

from .dataset import (
    AnnotationDataset,
    get_points_sampler,
    get_train_augmentator,
    get_val_augmentator,
)

logger = logging.getLogger(__name__)


COMMAND_DESCRIPTION = _(
    "Run finetune trains using a dataset in the format the annotator generates"
)


def command(parser):
    parser.add_argument(
        "model_path", type=Path, help=_("Path to the model script.")
    )
    parser.add_argument(
        "images_path", type=Path, help=_("Path to the dataset images.")
    )
    parser.add_argument(
        "masks_path", type=Path, help=_("Path to the dataset masks.")
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="experiment_path",
        type=Path,
        default=Path("."),
        help=_("Where to store experiment data"),
    )

    parser.add_argument(
        "-n",
        "--num-epochs",
        dest="num_epochs",
        type=int,
        help=_("Amount of epochs"),
    )

    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help=_(
            "Here you can specify the name of the experiment. It will be added as a suffix to the experiment folder."
        ),
    )  # noqa:E501

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help=_("Dataloader threads."),
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        type=int,
        default=-1,
        help=_(
            "You can override model batch size by specify positive number."
        ),
    )  # noqa:E501

    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help=_(
            'Number of GPUs. If you only specify "--gpus" argument, the ngpus value will be calculated automatically. You should use either this argument or "--gpus".'
        ),  # noqa:E501
    )  # noqa:E501

    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        required=False,
        help=_(
            'Ids of used GPUs. You should use either this argument or "--ngpus".'
        ),  # noqa: E501
    )  # noqa:E501

    parser.add_argument(
        "--resume-exp",
        type=str,
        default=None,
        help=_(
            'The prefix of the name of the experiment to be continued. If you use this field, you must specify the "--resume-prefix" argument.'
        ),  # noqa:E501
    )  # noqa:E501

    parser.add_argument(
        "--resume-prefix",
        type=str,
        default="latest",
        help=_("The prefix of the name of the checkpoint to be loaded."),
    )  # noqa:E501

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help=_(
            "The number of the starting epoch from which training will continue. (it is important for correct logging and learning rate)"
        ),
    )  # noqa:E501

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=_(
            "Model weights will be loaded from the specified path if you use this argument."
        ),  # noqa: E501
    )  # noqa:E501

    parser.add_argument(
        "--temp-model-path",
        type=str,
        default="",
        help=_("Do not use this argument (for internal purposes)."),
    )  # noqa:E501

    parser.add_argument(
        "--max-bigger-dimension",
        dest="max_bigger_dimension",
        type=int,
        help=_(
            "Resize the input dataset so the bigger dimension is this value"
        ),
    )

    parser.add_argument("--local_rank", type=int, default=0)

    def handle(args):
        model_path = Path(args.model_path)
        if args.temp_model_path != "":
            logger.debug(
                _(
                    "Falling back to temp_model_path because model_path wasn't specified"
                )  # noqa:E501
            )
            model_path = Path(args.temp_model_path)
        if not model_path.is_absolute():
            model_path = (
                Path(__file__).parent.parent.parent / "models" / model_path
            )
        model_path = model_path.resolve()
        args.model_path = model_path
        logger.debug(
            _("Final model path: '{model_path}'").format(model_path=model_path)
        )

        model_script = load_module(model_path)

        model_base_name = model_script.__dict__.get("MODEL_NAME")

        args.distributed = "WORLD_SIZE" in os.environ
        cfg = init_experiment(args, model_base_name)
        cfg.weights = args.weights
        cfg = load_cfg_from_env(cfg, os.environ)

        torch.backends.cudnn.benchmark = True
        torch.multiprocessing.set_sharing_strategy("file_system")
        logger.debug(_("Basic validations passed"))
        model, model_cfg = model_script.init_model(cfg)

        train_augmentator = get_train_augmentator(model_cfg)
        val_augmentator = get_val_augmentator(model_cfg)

        trainer = model_script.get_trainer(
            model, cfg, model_cfg, no_dataset=True
        )
        points_sampler = get_points_sampler(model_cfg)
        dataset_common_args = dict(
            images_path=args.images_path,
            masks_path=args.masks_path,
            keep_background_prob=0.05,
            min_object_area=1000,
            points_sampler=points_sampler,
        )
        if args.max_bigger_dimension is not None:
            dataset_common_args[
                "max_bigger_dimension"
            ] = args.max_bigger_dimension
        trainer.trainset = AnnotationDataset(
            split="train",
            augmentator=train_augmentator,
            **dataset_common_args,
        )
        trainer.valset = AnnotationDataset(
            split="val",
            augmentator=val_augmentator,
            **dataset_common_args,
        )
        if args.num_epochs is None:
            args.num_epochs = model_cfg.default_num_epochs
        trainer._before_needed_hook()
        assert trainer.lr_scheduler is not None
        trainer.run(num_epochs=args.num_epochs)

    return handle
