# MMAction2 Architecture Guide for Basketball_51

## Overview

This document catalogs the MMAction2 codebase structure, config system, pipeline components,
available architectures, and dataset preparation patterns — researched for training a custom
8-class basketball action recognition model on the Basketball_51 dataset.

**Key design choice:** The pipeline uses **TSN (Temporal Segment Network) with ResNet50**
for **pure video-level action classification** — no spatial localization (no bounding boxes)
and no temporal localization (no action boundaries). Each video produces a single
8-class prediction. This avoids any modification to MMAction2's base architecture;
the config at `configs/recognition/tsn/tsn_basketball51.py` inherits from the standard
Kinetics-400 template and overrides only dataset-specific settings.

---

## 1. Codebase Layout

```text
mmaction2_repo/
├── configs/              # All training configs (inheritance-based)
│   ├── _base_/
│   │   ├── models/       # Base model definitions (tsn_r50.py, etc.)
│   │   ├── schedules/    # Base training schedules (sgd_100e.py)
│   │   └── default_runtime.py  # Base runtime settings
│   └── recognition/      # Per-architecture config folders
│       ├── tsn/          # Temporal Segment Network
│       ├── tsm/          # Temporal Shift Module
│       ├── slowfast/     # SlowFast pathways
│       └── mvit/         # Masked Video Transformer
├── mmaction/
│   ├── datasets/         # Dataset classes & transforms
│   │   ├── base.py       # BaseDataset
│   │   ├── video_dataset.py  # VideoDataset (used for Basketball_51)
│   │   └── transforms/   # All pipeline transforms
│   │       ├── loading.py     # DecordInit, DecordDecode, VideoDecode
│   │       ├── processing.py  # Resize, CenterCrop, MultiScaleCrop, Flip, etc.
│   │       ├── formatting.py  # FormatShape, PackActionInputs
│   │       ├── wrappers.py    # Pipeline wrappers
│   │       └── pose_transforms.py, text_transforms.py
│   ├── models/           # Model definitions (heads, backbones, necks)
│   └── pipelines/        # Pipeline orchestration
├── tools/
│   ├── train.py          # Main training entrypoint
│   └── dist_train.sh     # Multi-GPU launcher
└── data/                 # Dataset storage root
```

---

## 2. Config Inheritance System

Configs use a 3-level inheritance chain:

```
default_runtime.py (base runtime)
   ↓
schedules/sgd_100e.py (optimizer, LR schedule)
   ↓
models/tsn_r50.py (model architecture)
   ↓
recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py (task config)
```

Each child overrides only what differs. A custom config inherits from the task config,
then overrides `num_classes`, `dataset_type`, annotation paths, pipeline stages,
and training schedule parameters.

### Key config fields

- **model**: `cls_head.num_classes` — must match dataset class count
- **dataset**: `dataset_type`, `ann_file`, `data_root`, `pipeline`
- **dataloader**: `batch_size`, `num_workers`, sampler settings
- **train/val/test_cfg**: loop type (`EpochBasedTrainLoop`), `max_epochs`, interval
- **param_scheduler**: `MultiStepLR` milestones, gamma
- **optim_wrapper**: optimizer type, lr, momentum, weight_decay
- **load_from**: pre-trained checkpoint URL (from model zoo)

---

## 3. Pipeline Architecture (Transforms)

All pipelines are composed as ordered transform lists. The data flow is:

```text
1. DecordInit     — Opens video file with Decord backend
2. SampleFrames   — Samples frames for TSN: clip_len=1, interval=1, num_clips=3
3. DecordDecode   — Decodes selected frames
4. Resize(-1,256) — Aspect-ratio-preserving resize to short side 256
5. MultiScaleCrop — Random crop at multiple scales for augmentation
6. Resize(224)    — Resize to 224×224
7. Flip(0.5)      — Random horizontal flip
8. FormatShape    — Converts to NCHW (PyTorch expected format)
9. PackActionInputs — Final packing for model
```

**Val/Test pipelines differ only in sampling:**

- Val uses `CenterCrop` instead of `MultiScaleCrop` + `Flip`
- Test uses `TenCrop` for 10-crop evaluation + 25 clips for full coverage

### Transform Classes (mmaction/datasets/transforms/)

| Module | Key Classes | Purpose |
| -------- | ------------- | --------- |
| `loading.py` | `DecordInit`, `DecordDecode`, `VideoDecode`, `PytorchVideoInit`, `AudioDecode` | Video/audio loading |
| `processing.py` | `Resize`, `CenterCrop`, `MultiScaleCrop`, `Flip`, `RandomGaussianBlur`, `ColorJitter` | Spatial augmentation |
| `formatting.py` | `FormatShape`, `PackActionInputs`, `ToTensor` | Data format conversion |
| `wrappers.py` | `TorchVisionWrapper` | Bridge to torchvision transforms |

---

## 4. Available Architectures

### 4.1 TSN (Temporal Segment Network) — Config: `configs/recognition/tsn/`

- **Idea**: Divides video into segments, samples one clip per segment, aggregates predictions
- **Backbone**: ResNet50 (ImageNet-pretrained by default)
- **Input**: 1 frame × 3 clips per video
- **Best for**: Short-range action recognition, simple temporal patterns
- **Template**: `tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py`
- **Our choice**: Used as baseline for Basketball_51 (8 classes, ~10k videos)

### 4.2 TSM (Temporal Shift Module) — Config: `configs/recognition/tsm/`

- **Idea**: Shifts channels along temporal axis for efficient temporal mixing
- **Backbone**: ResNet50 with temporal shift modules inserted
- **Advantage**: Better temporal modeling than TSN with minimal parameter increase
- **Best for**: Medium-range temporal patterns (2-3 second clips)
- **Trade-off**: Slightly higher accuracy than TSN, similar speed

### 4.3 SlowFast — Config: `configs/recognition/slowfast/`

- **Idea**: Two pathways — Slow (low frame-rate, high channel count) captures spatial semantics; Fast (high frame-rate, low channel count) captures motion
- **Backbone**: 3D ResNet (Slow pathway) + low-channel 3D ResNet (Fast pathway)
- **Advantage**: Best accuracy for longer clips (>4 seconds), strong motion capture
- **Trade-off**: Much higher compute, 3D convolutions

### 4.4 MViT (Masked Video Transformer) — Config: `configs/recognition/mvit/`

- **Idea**: State-of-the-art video transformer with masked feature encoding
- **Backbone**: Multiscale Vision Transformer (pooling-based downsampling)
- **Advantage**: Highest accuracy, no 3D convolutions, good speed-accuracy Pareto
- **Trade-off**: Transformer scaling, larger memory footprint

### 4.5 Architecture Comparison Summary

| Architecture | Temporal Modeling | Accuracy Tier | Compute | Best Use |
|-------------|------------------|---------------|-------- |----------|
| **TSN**     | Segment pooling   | Baseline      | Low     | Simple actions, short clips |
| **TSM**     | Channel shift     | Good          | Low+    | Medium temporal patterns |
| **SlowFast**| Dual-pathway 3D  | Very good     | High    | Long clips, motion-heavy |
| **MViT**    | Transformer       | State-of-art  | Medium  | Complex actions |

**Recommendation for Basketball_51**: Start with **TSN** (fast, well-suited for short basketball clips). If accuracy plateaus, try **TSM** for better temporal modeling or **SlowFast** for motion-heavy sequences. MViT if dataset grows significantly.

---

## 5. Dataset Details — Basketball_51

| Class | Label | Meaning    | Videos |
|-------|-------|------------|--------|
| 2p0   | 0     | 2-pt miss  | 1,418  |
| 2p1   | 1     | 2-pt made  | 1,950  |
| 3p0   | 2     | 3-pt miss  | 2,132  |
| 3p1   | 3     | 3-pt made  | 1,178  |
| ft0   | 4     | FT miss    | 566    |
| ft1   | 5     | FT made    | 1,724  |
| mp0   | 6     | Misc miss  | 785    |
| mp1   | 7     | Misc made  | 558    |

**Total:** 10,311 videos | **Classes:** 8 | **Class imbalance:** Present (566 ft0 vs 2,132 3p0).
Consider weighted sampling or class-balanced loss if training shows bias.

---

## 6. Data Preparation

### Annotation Format

MMAction2 `VideoDataset` expects a simple text file:

```text
relative/path/to/video.mp4  <label_int>
```

Example for Basketball_51:

```text
2p0/2p0_v108_000437_x264.mp4 0
2p1/2p1_v108_000340_x264.mp4 1
3p0/3p0_v108_000001_x264.mp4 2
...
```

### Directory Structure

```text
data/basketball51/
├── videos_train/         # Training videos (80% split)
│   ├── 2p0/              → class 0 videos
│   ├── 2p1/              → class 1 videos
│   ├── 3p0/              → class 2 videos
│   ├── 3p1/              → class 3 videos
│   ├── ft0/              → class 4 videos
│   ├── ft1/              → class 5 videos
│   ├── mp0/              → class 6 videos
│   └── mp1/              → class 7 videos
├── videos_val/           # Validation videos (20% split)
│   ├── 2p0/ ...
│   └── ...
├── basketball51_train.txt  # Annotation: video_path  label
└── basketball51_val.txt    # Annotation: video_path  label
```

---

## 7. Training Configuration (TSN)

Key modifications from Kinetics-400 template:

| Parameter            | Kinetics-400      | Basketball_51      | Reason |
|----------------------|-------------------|--------------------|--------|
| `num_classes`        | 400               | 8                  | 8 action classes |
| `max_epochs`         | 100               | 50                 | Smaller dataset |
| `learning_rate`      | 0.01              | 0.005              | Fine-tuning |
| `milestones`         | [30, 60, 80]      | [20, 40]           | Shorter schedule |
| `dataset_type`       | 'VideoDataset'    | 'VideoDataset'     | Same format |
| `ann_file`           | kinetics400       | basketball51       | Custom annotations |
| `load_from`          | kinetics400 ckpt  | kinetics400 ckpt   | Transfer learning |

---

## 8. Evaluation

MMAction2 uses `AccMetric` evaluator, which computes top-1 and top-5 accuracy.
For Basketball_51 (8 classes), top-1 is the primary metric. Class-wise accuracy is
recommended due to class imbalance.

---

## 9. Quick-Start Script

A Python automation script (`quick_start.py`) can:

1. Scan Basketball_51 directory tree
2. Create 80/20 train/val splits with annotation files
3. Generate the TSN config file
4. Start training via `tools/train.py`
5. Optionally generate TSM or SlowFast configs for comparison

---

## 10. References

- MMAction2 docs: <https://github.com/open-mmlab/mmaction2>
- Finetune guide: docs/en/user_guides/finetune.md
- Dataset prep: docs/en/user_guides/prepare_dataset.md
- Custom dataset: docs/en/advanced_guides/customize_dataset.md
- Train/test: docs/en/user_guides/train_test.md
