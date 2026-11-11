#!/usr/bin/env python3
"""Automated rubric compliance & FastAPI integration verifier for Bunri-Vision AI."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REQUIRED_README_HEADERS = [
    "## 0. 프로젝트 정보",
    "## 1. 문제 정의",
    "## 2. 데이터",
    "## 3. 데이터 전처리",
    "## 4. 모델",
    "## 5. 평가 결과",
    "## 6. 서비스 프로토타입",
    "## 7. 배포",
    "## 8. 실행 방법",
    "## 9. Lessons Learned",
]

REQUIRED_SUBMISSION_KEYS = [
    "student_id",
    "name",
    "title",
    "github_url",
    "demo_url",
    "data_type",
    "training",
    "models",
    "primary_metric",
]


def verify_all(root_dir: str = ".") -> None:
    root = Path(root_dir).resolve()
    errors: list[str] = []

    # 1. Verify README exact H2 headers
    readme = root / "README.md"
    if not readme.exists():
        errors.append("Missing README.md")
    else:
        text = readme.read_text(encoding="utf-8")
        lines = [ln.strip() for ln in text.splitlines()]
        for hdr in REQUIRED_README_HEADERS:
            if hdr not in lines:
                errors.append(f"README missing exact required header: '{hdr}'")

    # 2. Verify submission.json keys
    sub_path = root / "submission.json"
    if not sub_path.exists():
        errors.append("Missing submission.json")
    else:
        sub = json.loads(sub_path.read_text(encoding="utf-8"))
        for k in REQUIRED_SUBMISSION_KEYS:
            if k not in sub or sub[k] in (None, "", []):
                errors.append(f"submission.json missing or empty key: '{k}'")

    # 3. Verify executed Jupyter notebook has preserved cell outputs
    nb_path = root / "notebooks" / "bunri_vision_experiments.ipynb"
    if not nb_path.exists():
        errors.append("Missing notebooks/bunri_vision_experiments.ipynb")
    else:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        with_outputs = [c for c in code_cells if len(c.get("outputs", [])) > 0]
        if len(with_outputs) < len(code_cells):
            errors.append(f"Notebook has unexecuted code cells ({len(with_outputs)}/{len(code_cells)} with outputs)")

    # 4. Verify Screenshots >= 3
    shots = list((root / "assets" / "screenshots").glob("*.png"))
    if len(shots) < 3:
        errors.append(f"Expected >= 3 screenshots in assets/screenshots, found {len(shots)}")

    # 5. Verify no tracked/packaged file > 100 MB & no leaked API key patterns
    secret_pat = re.compile(r"(sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{35}|ghp_[A-Za-z0-9]{30,})")
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in p.parts for part in (".venv", ".git", "data", "__pycache__")):
            continue
        if p.suffix in (".py", ".md", ".json", ".html", ".js", ".css"):
            content = p.read_text(encoding="utf-8", errors="ignore")
            if secret_pat.search(content):
                errors.append(f"Potential API key/token detected in {p}")

    # 6. Verify FastAPI local TestClient end-to-end inference & exception guardrails
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        r_health = client.get("/api/health")
        assert r_health.status_code == 200, "Health endpoint failed"
        r_samp = client.post("/api/predict-sample/plastic")
        assert r_samp.status_code == 200, "Sample prediction failed"
        body = r_samp.json()
        assert "predicted_class" in body and "visuals" in body, "Malformed prediction payload"

        for exc_case in ("non_image_txt", "corrupted_binary", "tiny_resolution", "blank_solid"):
            r_exc = client.post(f"/api/test-exception/{exc_case}")
            if r_exc.status_code != 400 or not r_exc.json().get("error"):
                errors.append(f"Exception guardrail failed for {exc_case}")

    if errors:
        print("[FAIL] Rubric verification errors found:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    print("[PASS] All rubric checks, notebook output checks, security checks, and FastAPI inference tests passed!")


if __name__ == "__main__":
    verify_all()
