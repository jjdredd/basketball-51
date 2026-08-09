# Training & Testing Guide — Basketball_51

## Overview

This pipeline uses **TSN (Temporal Segment Network)** with a ResNet50 backbone for
**video-level action classification** — no spatial detection (no bounding boxes) and
no temporal localization (no start/end boundaries). Each video gets a single
8-class prediction: 2pt made/miss, 3pt made/miss, FT made/miss, misc made/miss.

The config lives at `mmaction2/configs/recognition/tsn/tsn_basketball51.py` —
a standalone file that inherits from the Kinetics-400 template and overrides
for 8 classes.

---

## Prerequisites

```bash
cd /home/agentslop/slop/mmaction/basketball51-project
source venv/bin/activate
```

All dependencies installed (see [DEPENDENCIES.md](DEPENDENCIES.md)).
Dataset at `/home/agentslop/slop/mmaction/Basketball_51 dataset/`.
MMAction2 submodule checked out at `mmaction2/`.

---

## Step 1: Generate annotation files

The script scans the dataset directories, creates an 80/20 train/val split,
and writes annotation files:

```bash
python3 quick_start.py --step annotations
```

Or manually using the script as a module:

```bash
python3 -c "from quick_start import create_annotations; create_annotations()"
```

This creates:

- `mmaction2/data/basketball51/basketball51_train.txt` — ~80% of videos
- `mmaction2/data/basketball51/basketball51_val.txt` — ~20% of videos

Each annotation line: `class_dir/video.mp4  label_int`

---

## Step 2: Symlink videos into MMAction2 data directory

```bash
python3 quick_start.py --step symlink
```

Or manually:

```bash
cd /home/agentslop/slop/mmaction/basketball51-project
mkdir -p mmaction2/data/basketball51/videos_train mmaction2/data/basketball51/videos_val
```

Expected structure after symlinking:

```text
mmaction2/data/basketball51/
├── basketball51_train.txt       # train annotations
├── basketball51_val.txt         # val annotations
├── videos_train/                # training videos
│   ├── 2p0/                     → class 0 (2-pt miss)
│   ├── 2p1/                     → class 1 (2-pt made)
│   ├── 3p0/                     → class 2 (3-pt miss)
│   ├── 3p1/                     → class 3 (3-pt made)
│   ├── ft0/                     → class 4 (FT miss)
│   ├── ft1/                     → class 5 (FT made)
│   ├── mp0/                     → class 6 (misc miss)
│   └── mp1/                     → class 7 (misc made)
├── videos_val/                  # validation videos (same class structure)
```

---

## Step 3: Train the model

The config is already created at `mmaction2/configs/recognition/tsn/tsn_basketball51.py`.
It inherits from:

```text
../../_base_/models/tsn_r50.py       # TSN + ResNet50 backbone
../../_base_/schedules/sgd_100e.py   # SGD optimizer + schedule
../../_base_/default_runtime.py      # runtime defaults
```

Key Basketball_51 overrides: 8 classes, 50 epochs, LR 0.005, milestones [20, 40].

To start training:

```bash
cd /home/agentslop/slop/mmaction/basketball51-project/mmaction2

python3 tools/train.py configs/recognition/tsn/tsn_basketball51.py \
    --work-dir work_dirs/tsn_basketball51 \
    --seed 42 \
    --deterministic
```

The `quick_start.py` script does all steps at once:

```bash
cd /home/agentslop/slop/mmaction/basketball51-project
python3 quick_start.py
```

---

## Step 4: Monitor training

```bash
# View live loss and accuracy curves
tensorboard --logdir mmaction2/work_dirs/tsn_basketball51/tf_logs

# Check checkpoints
ls mmaction2/work_dirs/tsn_basketball51/
```

Checkpoints are saved every 3 epochs as `epoch_{N}.pth`. The best model is
available in the work directory.

---

## Step 5: Test / evaluate

```bash
cd /home/agentslop/slop/mmaction/basketball51-project/mmaction2

python3 tools/test.py configs/recognition/tsn/tsn_basketball51.py \
    work_dirs/tsn_basketball51/best.pth \
    --dump work_dirs/tsn_basketball51/results.pkl \
    --eval top_k_accuracy 2
```

**Evaluation metrics:**

- `top_k_accuracy` — top-1 and top-2 accuracy
- `mean_class_wise_accuracy` — per-class accuracy averaged
- `confusion_matrix` — detailed class prediction matrix

---

## Step 6: Run inference on new videos

```python
from mmaction.apis import init_recognizer, inference_recognizer

# Load trained model
model = init_recognizer(
    'configs/recognition/tsn/tsn_basketball51.py',
    'work_dirs/tsn_basketball51/best.pth')

# Predict a new video
result = inference_recognizer(model, 'path/to/new_video.mp4')
pred_class = result['pred_score'].argmax()
print(f'Predicted class: {pred_class}')
```

---

## Customising the config

| Parameter | Default | Notes |
| ----------- | --------- | ------- |
| `batch_size` | 32 | Adjust for GPU memory (8 for 8GB, 32 for 24GB+) |
| `max_epochs` | 50 | Increase for underfitting, decrease for overfitting |
| `load_from` | Kinetics-400 ckpt | Change to a custom checkpoint or set to None for scratch |
| `milestones` | [20, 40] | LR drop epochs — adjust for longer/shorter training |

Edit `mmaction2/configs/recognition/tsn/tsn_basketball51.py` directly and re-run training.
