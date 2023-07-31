from pathlib import Path
from tempfile import mkdtemp

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

    assert type(model_script.MODEL_NAME) == str
    assert (
        model_script.__dict__.get("main") is None
    ), "Remove the main function"

    testing_tmpdir = mkdtemp()
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
    cfg.IMAGENET_PRETRAINED_MODELS = edict()
    cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18 = testing_tmpdir
    cfg.PASCALVOC_PATH = testing_tmpdir
    cfg.LVIS_PATH = testing_tmpdir
    cfg.LVIS_v1_PATH = testing_tmpdir
    cfg.COCO_PATH = testing_tmpdir
    cfg.SBD_PATH = testing_tmpdir
    cfg.OPENIMAGES_PATH = testing_tmpdir
    cfg.ADE20K_PATH = testing_tmpdir

    model, model_cfg = model_script.init_model(cfg, dry_run=True)
    trainer = model_script.get_trainer(model, cfg, model_cfg, dry_run=True)

    assert type(model_cfg.default_num_epochs) == int
    assert type(trainer) == ISTrainer
