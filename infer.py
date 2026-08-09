#!/usr/bin/env python3
"""Basketball_51 — single-video inference script.

Usage:
    python infer.py <video_path>              # GPU (default)
    python infer.py <video_path> --cpu        # CPU-only
    python infer.py <video_path> -o output    # save frame overlay
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
MMACTION_DIR = PROJECT_DIR / "mmaction2"
CONFIG_PATH = MMACTION_DIR / "configs" / "recognition" / "tsn" / "tsn_basketball51.py"
CHECKPOINT = MMACTION_DIR / "work_dirs" / "tsn_basketball51" / "best_acc_top1_epoch_41_fixed.pth"

# 8-class label map (must match config)
CLASS_MAP = {
    0: "2-pt miss",
    1: "2-pt made",
    2: "3-pt miss",
    3: "3-pt made",
    4: "FT miss",
    5: "FT made",
    6: "Misc miss",
    7: "Misc made",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basketball_51 video inference")
    parser.add_argument("video", type=str, help="Path to video file (.mp4)")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on CPU instead of GPU",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional output path for a frame overlay image",
    )
    return parser.parse_args()


def infer_video(video_path: str, use_cpu: bool = False) -> dict:
    """Run MMAction2 inference on a single video.

    Returns a dict with predicted class, confidence, and per-class scores.
    """
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Device selection
    if use_cpu:
        device = "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logging.warning("GPU requested but CUDA unavailable — falling back to CPU")

    from mmaction.apis import init_recognizer, inference_recognizer
    

    logging.info("Loading config: %s", CONFIG_PATH)
    logging.info("Using device: %s", device)
    logging.info("Checkpoint: %s", CHECKPOINT)

    # Build model and load checkpoint
    model = init_recognizer(str(CONFIG_PATH), str(CHECKPOINT), device=device)
    model.eval()

    # Run inference
    logging.info("Running inference on: %s", video_path)
    data_sample = inference_recognizer(model, video_path)
    scores = data_sample.pred_score.cpu().numpy().flatten()

    # Decode results
    top_idx = scores.argmax()
    top_confidence = float(scores[top_idx])
    top_class = int(top_idx)
    top_label = CLASS_MAP.get(top_class, f"class-{top_class}")

    # Top-k sorted scores
    topk = 2
    sorted_idx = scores.argsort()[::-1][:topk]
    topk_results = [
        {
            "class": int(idx),
            "label": CLASS_MAP.get(int(idx), f"class-{int(idx)}"),
            "confidence": float(scores[idx]),
        }
        for idx in sorted_idx
    ]

    result = {
        "video": video_path,
        "predicted_class": top_class,
        "predicted_label": top_label,
        "confidence": top_confidence,
        "topk": topk_results,
        "all_scores": {CLASS_MAP.get(i, f"class-{i}"): float(scores[i]) for i in range(len(scores))},
        "device": device,
    }

    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    args = parse_args()
    video_path = args.video
    use_cpu = args.cpu

    result = infer_video(video_path, use_cpu)

    # Print structured output
    print("\n" + "=" * 60)
    print(f"  Inference Result")
    print("=" * 60)
    print(f"  Video:      {result['video']}")
    print(f"  Device:     {result['device']}")
    print(f"  Predicted:  {result['predicted_label']} (class {result['predicted_class']})")
    print(f"  Confidence: {result['confidence']:.4f}")
    print()
    print("  Top-2 predictions:")
    for i, r in enumerate(result["topk"], 1):
        print(f"    {i}. {r['label']} — {r['confidence']:.4f}")
    print()
    print("  Per-class scores:")
    for label, score in result["all_scores"].items():
        print(f"    {label:>12s}: {score:.4f}")
    print("=" * 60)

    # Save JSON output alongside video
    out_json = Path(video_path).with_suffix(".infer.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    logging.info("Saved results to: %s", out_json)


if __name__ == "__main__":
    main()
