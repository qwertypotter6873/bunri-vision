#!/usr/bin/env python3
"""Download trained model weights from GitHub Releases (v1.0.0) if not present locally."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

RELEASE_BASE = "https://github.com/qwertypotter6873/bunri-vision/releases/download/v1.0.0"

WEIGHT_FILES = {
    "production_int8": "bunri_convnext_tiny_int8.onnx",
    "m1_scratch": "m1_scratch_cnn_fp16.pt",
    "m2_mobilenet": "m2_mobilenetv3_fp16.pt",
    "m3_resnet50": "m3_resnet50_fp16.pt",
    "m4_ablation": "m4_ablation_raw_fp16.pt",
    "m4_proposed": "m4_proposed_convnext_fp16.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Bunri-Vision AI trained model checkpoints.")
    parser.add_argument("--all", action="store_true", help="Download all 5 PyTorch/ONNX checkpoints (default: production ONNX)")
    parser.add_argument("--models-dir", default="models", help="Target directory for model weights")
    args = parser.parse_args()

    out_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = list(WEIGHT_FILES.values()) if args.all else [WEIGHT_FILES["production_int8"]]

    for fname in targets:
        dest = out_dir / fname
        if dest.exists():
            print(f"[SKIP] Already exists: {dest} ({dest.stat().st_size / (1024**2):.2f} MB)")
            continue
        url = f"{RELEASE_BASE}/{fname}"
        print(f"[DOWNLOAD] {url} -> {dest} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[OK] Saved {dest} ({dest.stat().st_size / (1024**2):.2f} MB)")


if __name__ == "__main__":
    main()
