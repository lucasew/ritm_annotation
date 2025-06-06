# flake8: noqa

from ritm_annotation.utils.exp_imports.default import *

MODEL_NAME = "resnet34"


def init_model(cfg, dry_run=False):
    model_cfg = edict()
    model_cfg.crop_size = (320, 480)
    model_cfg.num_max_points = 24
    model_cfg.default_num_epochs = 140

    model = DeeplabModel(
        backbone="resnet34",
        deeplab_ch=128,
        aspp_dropout=0.20,
        use_leaky_relu=True,
        use_rgb_conv=False,
        use_disks=True,
        norm_radius=5,
    )

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type="gaussian", magnitude=2.0))
    if not dry_run:
        model.feature_extractor.load_pretrained_weights()

    return model, model_cfg


def get_trainer(model, cfg, model_cfg, dry_run=False, no_dataset=False):
    cfg.batch_size = 28 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    train_augmentator = Compose(
        [
            UniformRandomResize(scale_range=(0.75, 1.40)),
            HorizontalFlip(),
            PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
            RandomCrop(*crop_size),
            RandomBrightnessContrast(
                brightness_limit=(-0.25, 0.25),
                contrast_limit=(-0.15, 0.4),
                p=0.75,
            ),
            RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
        ],
        p=1.0,
    )

    val_augmentator = Compose(
        [
            PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
            RandomCrop(*crop_size),
        ],
        p=1.0,
    )

    points_sampler = MultiPointSampler(
        model_cfg.num_max_points,
        prob_gamma=0.8,
        merge_objects_prob=0.15,
        max_num_merged_objects=2,
    )

    if no_dataset:
        trainset = None
        valset = None
    else:
        trainset = OpenImagesDataset(
            cfg.OPENIMAGES_PATH,
            split="train",
            augmentator=train_augmentator,
            min_object_area=1000,
            keep_background_prob=0.05,
            points_sampler=points_sampler,
            epoch_len=30000,
            dry_run=dry_run,
        )

        valset = OpenImagesDataset(
            cfg.OPENIMAGES_PATH,
            split="val",
            augmentator=val_augmentator,
            min_object_area=1000,
            keep_background_prob=0.05,
            points_sampler=points_sampler,
            epoch_len=2000,
            dry_run=dry_run,
        )

    optimizer_params = {"lr": 5e-4, "betas": (0.9, 0.999), "eps": 1e-8}

    lr_scheduler = partial(
        torch.optim.lr_scheduler.MultiStepLR, milestones=[120, 135], gamma=0.1
    )
    return ISTrainer(
        model,
        cfg,
        model_cfg,
        loss_cfg,
        trainset,
        valset,
        optimizer="adam",
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=5,
        image_dump_interval=2000,
        metrics=[AdaptiveIoU()],
        max_interactive_points=model_cfg.num_max_points,
        dry_run=dry_run,
    )
