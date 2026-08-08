# Training & Testing Guide — Basketball_51

## Prerequisites

- Activate the venv: `source venv/bin/activate`
- All dependencies installed (see [DEPENDENCIES.md](DEPENDENCIES.md))
- Dataset at `/home/agentslop/slop/mmaction/Basketball_51 dataset/`
- MMAction2 submodule checked out at `mmaction2/`

---

## Step 1: Create annotation files

Generate train/val `.txt` annotation files from the dataset directory listing:

```bash
cd /home/agentslop/slop/mmaction/basketball51-project

python3 quick_start.py --step annotations
```

This creates:

- `data/basketball51/train.txt`  — ~80% of videos
- `data/basketball51/val.txt`    — ~20% of videos

Each line: `relative_video_path label`

---

## Step 2: Structure the video data

Symlink (or copy) class directories into the expected data layout:

```bash
# Option A — symlinks (fast, zero copy)
mkdir -p data/basketball51
ln -s /home/agentslop/slop/mmaction/Basketball_51\ dataset/{Baseketball,Football,...} data/basketball51/

# Option B — copy (safe, independent copy)
cp -r /home/agentslop/slop/mmaction/Basketball_51\ dataset/* data/basketball51/
```

Expected structure:

```
data/basketball51/
├── Baseketball/     (class 0)
│   ├── vid_001.mp4
│   ├── vid_002.mp4
│   └── ...
├── Football/        (class 1)
├── ...
├── train.txt
├── val.txt
└── annotations/
    ├── train.txt
    └── val.txt
```

The `quick_start.py` script does this automatically:

```bash
python3 quick_start.py --step symlink
```

---

## Step 3: Create the config file

The quick-start script generates `configs/tsn_basketball51.py` with:

- 8-class head (instead of Kinetics-400's 400)
- Data root pointing to `data/basketball51/`
- TSN backbone (ResNet50, 8-frame input)
- SGD optimizer with 100-epoch schedule
- Standard pipeline (crop, flip, normalization)

Generate it:

```bash
python3 quick_start.py --step config
```

Or manually review the generated config at `configs/tsn_basketball51.py`.

---

## Step 4: Train the model

```bash
cd /home/agentslop/slop/mmaction/basketball51-project/mmaction2

python3 tools/train.py ../configs/tsn_basketball51.py \
    --work-dir ../work_dirs/tsn_basketball51 \
    --auto-resume \
    --auto-scale-lr \
    --cfg-options \
        "dataset_type=VideoDataset" \
        "data.train.ann_file=../data/basketball51/train.txt" \
        "data.val.ann_file=../data/basketball51/val.txt" \
        "data.train.data_prefix=../data/basketball51/" \
        "data.val.data_prefix=../data/basketball51/" \
        "train_dataloader.batch_size=16" \
        "val_dataloader.batch_size=16"
```

**Key flags:**

- `--work-dir` — where checkpoints and logs are saved
- `--auto-resume` — pick up from the latest checkpoint if interrupted
- `--auto-scale-lr` — adjust learning rate proportionally to batch size
- `--cfg-options` — override config values at runtime

---

## Step 5: Monitor training

```bash
# View live loss curves
tensorboard --logdir ../work_dirs/tsn_basketball51/tf_logs

# Check latest checkpoint
ls ../work_dirs/tsn_basketball51/
```

Checkpoints are saved every epoch as `epoch_{N}.pth`. The best model is symlinked as `best.pth`.

---

## Step 6: Test/evaluate the model

```bash
cd /home/agentslop/slop/mmaction/basketball51-project/mmaction2

python3 tools/test.py ../configs/tsn_basketball51.py \
    ../work_dirs/tsn_basketball51/best.pth \
    --dump ../work_dirs/tsn_basketball51/results.pkl \
    --cfg-options \
        "data.val.ann_file=../data/basketball51/val.txt" \
        "data.val.data_prefix=../data/basketball51/" \
    --eval top_k_accuracy 2
```

**Evaluation metrics:**

- `top_k_accuracy` — top-1 and top-2 accuracy
- `mean_class_wise_accuracy` — per-class accuracy averaged
- `confusion_matrix` — detailed class prediction matrix

---

## Quick-start (all steps at once)

```bash
cd /home/agentslop/slop/mmaction/basketball51-project

# 1. Create annotations + symlink videos + generate config
python3 quick_start.py --step all

# 2. Start training
cd mmaction2
python3 tools/train.py ../configs/tsn_basketball51.py \
    --work-dir ../work_dirs/tsn_basketball51
```

---

## Customising the config

| Parameter | Default | Notes |
| ----------- | --------- | ------- |
| `batch_size` | 16 | Adjust for GPU memory (8 for 8GB, 32 for 24GB+) |
| `max_epochs` | 100 | Increase for underfitting, decrease for overfitting |
| `workers_per_dataloader` | 4 | Reduce if CPU I/O is a bottleneck |
| `standard_aug` | `True` | Disable for small datasets to avoid over-augmentation |
| `init_weights` | `True` | Load ImageNet-pretrained ResNet50 |

Edit `configs/tsn_basketball51.py` directly or override via `--cfg-options`.
