#!/usr/bin/env python3
"""
Fix checkpoint for PyTorch 2.6+ compatibility.
Removes non-tensor objects (message_hub) so torch.load with weights_only=True works.
Usage:
    python fix_checkpoint.py <input_checkpoint> [output_checkpoint]
"""
import argparse
import torch
import os


def fix_checkpoint(input_path: str, output_path: str | None = None) -> str:
    """Load checkpoint with weights_only=False, strip non-tensor objects, re-save."""
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_fixed{ext}"

    ckpt = torch.load(input_path, weights_only=False, map_location="cpu")

    # Keep only state_dict (model weights) and meta (metadata).
    # message_hub contains HistoryBuffer which is not safe by default in PyTorch 2.6+.
    fixed = {
        "state_dict": ckpt["state_dict"],
        "meta": ckpt.get("meta", {}),
    }

    torch.save(fixed, output_path)
    print(f"Fixed checkpoint saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fix MMAction2 checkpoint for PyTorch 2.6+"
    )
    parser.add_argument("input", help="Path to input checkpoint (.pth)")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path (default: <input>_fixed.pth)",
    )
    args = parser.parse_args()
    fix_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
