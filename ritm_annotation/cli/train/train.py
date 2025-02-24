import logging
import os
from gettext import gettext as _
from pathlib import Path

import torch

from ritm_annotation.utils.env import load_cfg_from_env
from ritm_annotation.utils.exp import init_experiment
from ritm_annotation.utils.misc import load_module

logger = logging.getLogger(__name__)


def handle(args):
    model_path = Path(args.model_path)
    if args.temp_model_path != "":
        logger.debug(
            _("Falling back to temp_model_path because model_path wasn't specified")  # noqa:E501
        )
        model_path = Path(args.temp_model_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent.parent.parent / "models" / model_path
    model_path = model_path.resolve()
    args.model_path = model_path
    logger.debug(_("Final model path: '{model_path}'").format(model_path=model_path))

    model_script = load_module(model_path)

    model_base_name = model_script.__dict__.get("MODEL_NAME")

    args.distributed = "WORLD_SIZE" in os.environ
    cfg = init_experiment(args, model_base_name)
    cfg = load_cfg_from_env(cfg, os.environ)

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy("file_system")
    logger.debug(_("Basic validations passed"))
    model, model_cfg = model_script.init_model(cfg)
    trainer = model_script.get_trainer(model, cfg, model_cfg)
    if args.num_epochs is None:
        args.num_epochs = model_cfg.default_num_epochs
    trainer.run(num_epochs=args.num_epochs)
