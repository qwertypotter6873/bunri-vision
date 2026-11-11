#!/usr/bin/env python3
"""Package the final CyberCampus submission archive: 기말-{학번}-{이름}.zip

Includes:
1. 포트폴리오_README.pdf (exact PDF rendering of README.md)
2. README.md
3. submission.json
4. Executed Jupyter Notebook (notebooks/bunri_vision_experiments.ipynb, output cells preserved)
5. Full Python source code (src/, app/, scripts/, requirements.txt, requirements-train.txt, Dockerfile)
6. Training logs & evaluation artifacts (logs/, assets/, sample_data/, models/model_metadata.json)

Note: Model binary weights >100MB (and 28MB ONNX) are hosted on GitHub Releases with direct links in
README.md & auto-download in app/inference.py + scripts/download_weights.py so the LMS zip stays compact (~12MB).
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def build_zip(root_dir: str = ".", out_dir: str = "..") -> Path:
    root = Path(root_dir).resolve()
    sub = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    student_id = sub["student_id"]
    name = sub["name"]

    zip_filename = f"기말-{student_id}-{name}.zip"
    zip_path = Path(out_dir).resolve() / zip_filename

    include_files: list[Path] = [
        root / "포트폴리오_README.pdf",
        root / "README.md",
        root / "submission.json",
        root / "requirements.txt",
        root / "requirements-train.txt",
        root / "Dockerfile",
        root / ".gitignore",
        root / "models" / "model_metadata.json",
    ]

    for folder in ["src", "app", "scripts", "notebooks", "logs", "assets", "sample_data"]:
        fdir = root / folder
        if fdir.exists():
            for p in sorted(fdir.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("."):
                    include_files.append(p)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fpath in include_files:
            if not fpath.exists():
                raise FileNotFoundError(f"Required submission file missing: {fpath}")
            if fpath.stat().st_size > 100 * 1024 * 1024:
                raise ValueError(f"File exceeds 100MB limit: {fpath}")
            arcname = str(fpath.relative_to(root))
            zf.write(fpath, arcname=arcname)

    size_mb = zip_path.stat().st_size / (1024**2)
    print(f"[OK] Built submission archive: {zip_path} ({size_mb:.2f} MB, {len(include_files)} files)")
    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="..")
    args = parser.parse_args()
    build_zip(args.root, args.out_dir)
