import os
from pathlib import Path


COMMAND_DESCRIPTION = _("Run from-scratch trains using off the shelf datasets")


def command(parser):
    parser.add_argument(
        "model_path", type=str, help=_("Path to the model script.")
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
        ),
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

    parser.add_argument("--local_rank", type=int, default=0)


    def handle(args):
        from .train import handle as train_handle
        train_handle(args)

    return handle
