import ritm_annotation.utils.i18n  # noqa:F401

from easydict import EasyDict as edict

from .env import load_cfg_from_env


def test_load_cfg_from_env():
    input_dict = {"RITM_a": 2, "RITM_eoq__trabson": 3}
    loaded = load_cfg_from_env(edict(), input_dict)
    assert loaded.a == 2
    assert loaded.eoq.trabson == 3
