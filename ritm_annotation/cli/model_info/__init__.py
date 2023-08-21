import logging
from pathlib import Path

import torch

from ritm_annotation.inference.utils import (
    find_checkpoint,
    load_single_is_model,
)

logger = logging.getLogger(__name__)

COMMAND_DESCRIPTION = _("Show information about a model pth file")


def command(subparser):
    subparser.add_argument("model", type=Path)

    def handle(args):
        logger.debug(_("Loading model"))
        assert args.model.exists() and args.model.is_file(), _(
            "Model must exist and be a file"
        )
        checkpoint_path = find_checkpoint(args.model.parent, args.model.name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        print(f"model {state_dict['config']['class']}")
        for k, v in state_dict["config"]["params"].items():
            print(f"config {k} {v['value']}")
        for k, v in state_dict["state_dict"].items():
            print(
                f"state {k} {str(v.dtype).replace('torch.', '')}[{tuple(v.shape)}]"  # noqa:E501
            )  # noqa:E501
        model = load_single_is_model(state_dict, torch.device("cpu"))
        for line in repr(model).split("\n"):
            print(f"arch {line.rstrip()}")

    return handle
