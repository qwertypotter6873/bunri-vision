#!/usr/bin/env python3
"""Assemble the static GitHub Pages bundle (_site/) for Bunri-Vision AI.

Copies the web UI, trained INT8 ONNX checkpoint, held-out sample images,
and benchmark metadata into a self-contained static directory with zero
file duplication in Git history.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_pages_site(root_dir: str = ".", out_dir: str = "_site") -> Path:
    root = Path(root_dir).resolve()
    site = (root / out_dir).resolve()
    if site.exists():
        shutil.rmtree(site)

    (site / "static").mkdir(parents=True, exist_ok=True)
    (site / "models").mkdir(parents=True, exist_ok=True)
    (site / "samples").mkdir(parents=True, exist_ok=True)
    (site / "data").mkdir(parents=True, exist_ok=True)

    # Disable Jekyll processing on GitHub Pages
    (site / ".nojekyll").write_text("", encoding="utf-8")

    # HTML & static frontend assets
    shutil.copy2(root / "app" / "static" / "index.html", site / "index.html")
    for fname in ("styles.css", "vision_engine.js", "app.js"):
        shutil.copy2(root / "app" / "static" / fname, site / "static" / fname)

    # Trained ONNX checkpoint & metadata
    shutil.copy2(
        root / "models" / "bunri_convnext_tiny_int8.onnx",
        site / "models" / "bunri_convnext_tiny_int8.onnx",
    )
    shutil.copy2(root / "models" / "model_metadata.json", site / "models" / "model_metadata.json")

    # Held-out test samples & benchmark JSON
    for p in sorted((root / "sample_data").glob("*.jpg")):
        shutil.copy2(p, site / "samples" / p.name)
    shutil.copy2(root / "sample_data" / "manifest.json", site / "data" / "samples_manifest.json")
    shutil.copy2(root / "logs" / "benchmark_results.json", site / "data" / "benchmark_results.json")

    total_mb = sum(p.stat().st_size for p in site.rglob("*") if p.is_file()) / (1024**2)
    print(f"[OK] Built GitHub Pages bundle at {site} ({total_mb:.2f} MB)")
    return site


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GitHub Pages static bundle.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="_site")
    args = parser.parse_args()
    build_pages_site(args.root, args.out)
