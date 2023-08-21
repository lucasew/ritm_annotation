import ritm_annotation.utils.i18n  # noqa:F401

from pathlib import Path

import pytest
from easydict import EasyDict as edict

from ritm_annotation.engine.trainer import ISTrainer
from ritm_annotation.utils.misc import load_module

all_models = []

for model in Path(__file__).parent.glob("**/*.py"):
    if model.name == "__init__.py":
        continue
    if model.name == "test_exposed_model_stuff.py":
        continue
    all_models.append(model)


@pytest.mark.parametrize(
    "model_file",
    all_models,
    ids=[f"{x.parent.name}/{x.name}" for x in all_models],
)
def test_model_exposes_the_right_stuff(model_file):
    model_script = load_module(model_file)

    assert isinstance(model_script.MODEL_NAME, str)
    assert model_script.__dict__.get("main") is None, _(
        "Remove the main function"
    )

    cfg = edict()
    cfg.device = "cpu"
    cfg.batch_size = -1  # use default
    cfg.distributed = False
    cfg.workers = 0  # it's dataloader stuff
    cfg.weights = None
    cfg.resume_exp = None
    cfg.multi_gpu = False
    cfg.local_rank = 0  # master
    cfg.start_epoch = 0

    model, model_cfg = model_script.init_model(cfg, dry_run=True)
    trainer = model_script.get_trainer(
        model, cfg, model_cfg, dry_run=True, no_dataset=True
    )

    assert isinstance(model_cfg.default_num_epochs, int)
    assert isinstance(trainer, ISTrainer)
    assert trainer.trainset is None  # test handling of no_dataset
    assert trainer.valset is None
