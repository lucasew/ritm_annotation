# ritm_annotation

[![codecov](https://codecov.io/gh/lucasew/ritm_annotation/branch/main/graph/badge.svg?token=ritm_annotation_token_here)](https://codecov.io/gh/lucasew/ritm_annotation)
[![CI](https://github.com/lucasew/ritm_annotation/actions/workflows/main.yml/badge.svg)](https://github.com/lucasew/ritm_annotation/actions/workflows/main.yml)

Tool to do dataset annotation for semantic segmentation datasets.

Based on [SamsungLabs/ritm_interactive_segmentation](https://github.com/SamsungLabs/ritm_interactive_segmentation)

**Work in progress**

## Install it from PyPI

```bash
pip install ritm_annotation
```

## Running it using Nix flakes
```
nix run github:lucasew/ritm_annotation -- --help
```

## Usage

```bash
$ python -m ritm_annotation --help
# or
$ ritm_annotation --help
```

## Pretrained model weights
[Here you can download pretrained weights](https://github.com/SamsungLabs/ritm_interactive_segmentation/releases/tag/v1.0)

In my tests I used `coco_lvis_h18_itermask.pth` (SHA256 5f69cfce354d1507e3850bfc39ee7057c8dd27b6a4910d1d2dc724916b9ee32b)


## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
