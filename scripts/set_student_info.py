#!/usr/bin/env python3
"""Update student_id & name across README.md, submission.json, PDF, and submission zip."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.build_submission_zip import build_zip
from scripts.render_portfolio_pdf import render_pdf


def update_info(student_id: str, name: str, root_dir: str = ".") -> None:
    root = Path(root_dir).resolve()
    sub_path = root / "submission.json"
    sub = json.loads(sub_path.read_text(encoding="utf-8"))
    old_id = sub["student_id"]
    old_name = sub["name"]
    sub["student_id"] = student_id
    sub["name"] = name
    sub_path.write_text(json.dumps(sub, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme_path = root / "README.md"
    txt = readme_path.read_text(encoding="utf-8")
    txt = txt.replace(old_id, student_id).replace(old_name, name)
    readme_path.write_text(txt, encoding="utf-8")

    render_pdf(str(readme_path), str(root / "포트폴리오_README.pdf"))
    build_zip(str(root), str(root.parent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update student_id & name and rebuild PDF + submission zip.")
    parser.add_argument("--student-id", required=True, help="Gachon student ID (e.g., 202135000)")
    parser.add_argument("--name", default="김대현", help="Student Korean name")
    args = parser.parse_args()
    update_info(args.student_id, args.name)
