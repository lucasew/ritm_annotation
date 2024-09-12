# flake8: noqa E501

import json
import logging
from gettext import gettext as _
from pathlib import Path
from sys import exit

COMMAND_DESCRIPTION = _("Interactively annotate a dataset")


def command(subparser):
    subparser.add_argument("input", type=Path)
    subparser.add_argument("output", type=Path)
    subparser.add_argument("-d", "--device", dest="device")
    subparser.add_argument(
        "-c", "--classes", dest="classes", type=str, required=True, nargs="+"
    )
    subparser.add_argument("-w", "--checkpoint", dest="checkpoint", type=Path)
    subparser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=Path,
        help=_("Folder with pre-defined segmentations to be adjusted"),
    )

    def handle(args):
        from .annotator import handle as annotator_handle

        annotator_handle(args)

    return handle
