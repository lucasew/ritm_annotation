import logging
from typing import Dict

from easydict import EasyDict as edict

logger = logging.getLogger(__name__)


def load_cfg_from_env(cfg: edict, env: Dict[str, str]):
    for k, v in env.items():
        if k.startswith("RITM_"):
            cfgkey = k.replace("RITM_", "").replace("__", ".")
            logger.warning(
                _(
                    "Changing configuration entry from environment variable: {k}={v}"
                ).format(
                    k=cfgkey, v=v
                )  # noqa:E501
            )  # noqa: E501
            *parts, last = cfgkey.split(".")
            this_cfg = cfg
            for part in parts:
                if this_cfg.get(part) is None:
                    this_cfg[part] = edict()
                this_cfg = this_cfg[part]
            this_cfg[last] = v
    return cfg
