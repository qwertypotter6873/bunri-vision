#!/usr/bin/env python3
"""Download and extract the official RealWaste dataset from the UCI ML Repository (#908)."""

from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

UCI_REALWASTE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"


def download_realwaste(data_dir: str = "data") -> Path:
    base = Path(data_dir)
    base.mkdir(parents=True, exist_ok=True)
    zip_path = base / "realwaste.zip"
    raw_dir = base / "raw"
    target_root = raw_dir / "realwaste-main" / "RealWaste"

    if target_root.exists() and any(target_root.iterdir()):
        print(f"[OK] RealWaste already extracted at: {target_root}")
        return target_root

    if not zip_path.exists():
        print(f"[INFO] Downloading RealWaste from {UCI_REALWASTE_URL} ...")
        urllib.request.urlretrieve(UCI_REALWASTE_URL, zip_path)
        print(f"[OK] Downloaded {zip_path} ({zip_path.stat().st_size / (1024**2):.1f} MB)")

    print(f"[INFO] Extracting {zip_path} -> {raw_dir} ...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    print(f"[OK] Extraction complete: {target_root}")
    return target_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download UCI RealWaste (#908) dataset.")
    parser.add_argument("--data-dir", default="data", help="Target directory for dataset storage")
    args = parser.parse_args()
    download_realwaste(args.data_dir)
