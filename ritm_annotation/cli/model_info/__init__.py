from gettext import gettext as _
from pathlib import Path

COMMAND_DESCRIPTION = _("Show information about a model pth file")


def command(subparser):
    subparser.add_argument("model", type=Path)

    def handle(args):
        from .model_info import handle as model_info_handle

        model_info_handle(args)

    return handle
