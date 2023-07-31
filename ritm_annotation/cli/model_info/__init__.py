from pathlib import Path
import logging

import torch

from ritm_annotation.inference.utils import find_checkpoint, load_is_model, load_single_is_model

logger = logging.getLogger(__name__)


def command(subparser):
    subparser.description = "Show information about a model pth"
    subparser.add_argument('model', type=Path)

    def handle(args):
        logger.debug('Loading model')
        assert args.model.exists() and args.model.is_file(), "Model must exist and be a file"
        checkpoint_path = find_checkpoint(args.model.parent, args.model.name)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model = load_single_is_model(state_dict, torch.device('cpu'))
        print(state_dict)
        print(model)
    return handle
