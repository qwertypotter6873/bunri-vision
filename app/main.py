"""FastAPI server for Bunri-Vision AI household waste classification & disposal guidance."""

from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.inference import BunriInferenceEngine
from src.domain_rules import CLASS_NAMES, KOREAN_WASTE_RULES
from src.preprocessing import InputValidationError, validate_and_decode_image

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLE_DIR = ROOT_DIR / "sample_data"
LOGS_DIR = ROOT_DIR / "logs"

engine: BunriInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = BunriInferenceEngine(root_dir=ROOT_DIR)
    yield


app = FastAPI(
    title="분리비전 AI (Bunri-Vision)",
    description="실환경 오염·혼합 생활폐기물 9종 비전 판독 및 한국형 분리배출 의사결정 서비스",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    assert engine is not None
    return {
        "status": "ok",
        "service": "Bunri-Vision AI",
        "loaded_checkpoint": engine.model_path.name,
        "checkpoint_size_mb": round(engine.model_path.stat().st_size / (1024**2), 2),
        "classes": CLASS_NAMES,
        "metadata": engine.metadata,
    }


@app.get("/api/rules")
def get_sorting_rules() -> Dict[str, Any]:
    return {
        "classes": [
            {
                "en_name": r.en_name,
                "ko_name": r.ko_name,
                "bin_category": r.bin_category,
                "bin_badge_type": r.bin_badge_type,
                "recyclable_status": r.recyclable_status,
                "steps": r.steps,
                "penalty_trap": r.penalty_trap,
                "examples": r.examples,
            }
            for r in KOREAN_WASTE_RULES.values()
        ]
    }


@app.get("/api/samples")
def list_samples() -> Dict[str, Any]:
    manifest_path = SAMPLE_DIR / "manifest.json"
    if not manifest_path.exists():
        return {"samples": []}
    with open(manifest_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        item["image_url"] = f"/samples/{item['filename']}"
    return {"samples": items}


@app.get("/api/benchmark")
def get_benchmark() -> Dict[str, Any]:
    bench_path = LOGS_DIR / "benchmark_results.json"
    if not bench_path.exists():
        return {"models": {}}
    with open(bench_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)) -> JSONResponse:
    assert engine is not None
    raw_bytes = await file.read()
    try:
        pil_img = validate_and_decode_image(
            raw_bytes=raw_bytes,
            filename=file.filename or "upload.jpg",
            content_type=file.content_type,
        )
    except InputValidationError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "error_code": exc.code,
                "message": exc.message,
            },
        )

    result = engine.predict(pil_img)
    return JSONResponse(content=result)


@app.post("/api/predict-sample/{sample_id}")
def predict_sample(sample_id: str) -> JSONResponse:
    assert engine is not None
    manifest_path = SAMPLE_DIR / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="샘플 매니페스트를 찾을 수 없습니다.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    matched = next((x for x in items if x["id"] == sample_id), None)
    if not matched:
        raise HTTPException(status_code=404, detail=f"존재하지 않는 샘플 ID: {sample_id}")

    img_path = SAMPLE_DIR / matched["filename"]
    pil_img = Image.open(img_path).convert("RGB")
    result = engine.predict(pil_img)
    result["sample_ground_truth"] = matched
    return JSONResponse(content=result)


@app.post("/api/test-exception/{case_type}")
def trigger_validation_exception(case_type: str) -> JSONResponse:
    """Built-in verifiable test cases demonstrating server-side input validation & error handling."""
    if case_type == "non_image_txt":
        raw_bytes, fname, ctype = b"Hello, this is a plain text note, not a waste photo!", "memo.txt", "text/plain"
    elif case_type == "corrupted_binary":
        raw_bytes, fname, ctype = b"\xff\xd8\xff\xe0\x00\x10BROKEN_JPEG_STREAM_XYZ", "broken_photo.jpg", "image/jpeg"
    elif case_type == "tiny_resolution":
        im = Image.new("RGB", (24, 24), (120, 160, 90))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        raw_bytes, fname, ctype = buf.getvalue(), "tiny_24x24.png", "image/png"
    elif case_type == "blank_solid":
        im = Image.new("RGB", (256, 256), (240, 240, 240))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        raw_bytes, fname, ctype = buf.getvalue(), "blank_wall.jpg", "image/jpeg"
    else:
        raise HTTPException(status_code=404, detail="알 수 없는 예외 테스트 유형입니다.")

    try:
        validate_and_decode_image(raw_bytes=raw_bytes, filename=fname, content_type=ctype)
    except InputValidationError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "error_code": exc.code,
                "tested_file": fname,
                "message": exc.message,
            },
        )
    return JSONResponse(content={"error": False})


if SAMPLE_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLE_DIR)), name="samples")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
