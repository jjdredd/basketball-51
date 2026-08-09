#!/usr/bin/env python3
"""Quick-start script for Basketball_51 training with MMAction2.

Steps:
  1. Prepare annotation files (train/val splits)
  2. Symlink videos into MMAction2 data directory
  3. Start training using the pre-built standalone config
"""

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
DATASET_DIR = Path("/home/agentslop/slop/mmaction/Basketball_51 dataset")
MMACTION_DIR = Path(__file__).resolve().parent / "mmaction2"
DATA_DIR = MMACTION_DIR / "data" / "basketball51"
TRAIN_DIR = DATA_DIR / "videos_train"
VAL_DIR = DATA_DIR / "videos_val"
TRAIN_ANNOT = DATA_DIR / "basketball51_train.txt"
VAL_ANNOT = DATA_DIR / "basketball51_val.txt"
CONFIG_PATH = (
    MMACTION_DIR
    / "configs"
    / "recognition"
    / "tsn"
    / "tsn_basketball51.py"
)
WORK_DIR = MMACTION_DIR / "work_dirs" / "tsn_basketball51"
TRAIN_SPLIT = 0.8  # 80% train, 20% val
RANDOM_SEED = 42

CLASS_LABELS = {
    "2p0": 0,  # 2-pt miss
    "2p1": 1,  # 2-pt made
    "3p0": 2,  # 3-pt miss
    "3p1": 3,  # 3-pt made
    "ft0": 4,  # FT miss
    "ft1": 5,  # FT made
    "mp0": 6,  # Misc miss
    "mp1": 7,  # Misc made
}


def create_annotations():
    """Create train/val annotation text files."""
    random.seed(RANDOM_SEED)
    print("📝 Creating annotation files...")

    # Collect all video paths with labels
    all_videos = []  # (relative_path, label)
    for class_dir, label in CLASS_LABELS.items():
        class_path = DATASET_DIR / class_dir
        if not class_path.is_dir():
            print(f"  ⚠️  Missing directory: {class_path}")
            continue
        try:
            entries = sorted(os.listdir(class_path))
        except (OSError, FileNotFoundError) as e:
            print(f"  ⚠️  Cannot list {class_path}: {e}")
            continue
        for fname in entries:
            if fname.endswith(".mp4"):
                all_videos.append((f"{class_dir}/{fname}", label))

    random.shuffle(all_videos)
    print(f"  Total videos: {len(all_videos)}")

    # Split
    try:
        split_idx = int(len(all_videos) * TRAIN_SPLIT)
    except (ValueError, TypeError) as e:
        print(f"  ⚠️  Cannot compute split index: {e}")
        raise
    train_videos = all_videos[:split_idx]
    val_videos = all_videos[split_idx:]

    # Write annotations
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, videos, out_file in [
        ("train", train_videos, TRAIN_ANNOT),
        ("val", val_videos, VAL_ANNOT),
    ]:
        try:
            with open(out_file, "w") as f:
                for rel_path, label in videos:
                    f.write(f"{rel_path} {label}\n")
        except (OSError, FileNotFoundError) as e:
            print(f"  ⚠️  Cannot write {out_file}: {e}")
            raise
        print(f"  {split_name}: {len(videos)} videos → {out_file.name}")

    return train_videos, val_videos


def symlink_videos(train_videos, val_videos):
    """Symlink videos into MMAction2 data directory."""
    print("\n🔗 Symlinking videos...")

    for split_name, videos, target_dir in [
        ("train", train_videos, TRAIN_DIR),
        ("val", val_videos, VAL_DIR),
    ]:
        for rel_path, _ in videos:
            src = DATASET_DIR / rel_path
            dst = target_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                # Path is constrained to DATASET_DIR — safe
                src_resolved = os.path.realpath(os.path.abspath(src))
                dst_resolved = os.path.realpath(os.path.abspath(dst))
                os.symlink(src_resolved, dst_resolved)
        print(f"  {split_name}: {len(videos)} symlinks created in {target_dir.name}")


def start_training():
    """Launch MMAction2 training using the standalone config."""
    print("\n🚀 Starting training...")
    print(f"  Config: {CONFIG_PATH}")
    print(f"  Work dir: {WORK_DIR}")
    cmd = [
        sys.executable or "python",
        str(MMACTION_DIR / "tools" / "train.py"),
        str(CONFIG_PATH),
        "--work-dir",
        str(WORK_DIR),
        "--seed",
        str(RANDOM_SEED),
        "--deterministic",
    ]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(MMACTION_DIR))
    return result.returncode


if __name__ == "__main__":
    print("=" * 60)
    print("  Basketball_51 — Quick-Start Training Script")
    print("=" * 60)

    # Step 1: Annotations
    train_vids, val_vids = create_annotations()

    # Step 2: Symlink videos
    symlink_videos(train_vids, val_vids)

    # Step 3: Train (config already exists at tsn_basketball51.py)
    rc = start_training()

    if rc == 0:
        print("\n✅ Training completed successfully!")
    else:
        print(f"\n❌ Training failed with exit code {rc}")
    sys.exit(rc)
