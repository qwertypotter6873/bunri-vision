"""Production ONNX Runtime inference engine + Spatial CAM visualizer for Bunri-Vision AI."""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from src.domain_rules import CLASS_NAMES, build_decision_advice
from src.preprocessing import preprocess_for_inference

DEFAULT_WEIGHT_Candidates = [
    "models/bunri_convnext_tiny_int8.onnx",
    "models/bunri_convnext_tiny.onnx",
]


def _encode_rgb_to_data_url(rgb_arr: np.ndarray, quality: int = 88) -> str:
    pil_img = Image.fromarray(rgb_arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _render_cam_overlay(rgb_224: np.ndarray, cam_7x7: np.ndarray, alpha: float = 0.44) -> np.ndarray:
    cam = np.maximum(cam_7x7, 0.0)
    cam = cam - cam.min()
    if cam.max() > 1e-6:
        cam = cam / cam.max()
    cam_224 = cv2.resize(cam.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)
    cam_224 = np.clip(cam_224, 0.0, 1.0)
    heatmap_bgr = cv2.applyColorMap((cam_224 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = (1.0 - alpha) * rgb_224.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


class BunriInferenceEngine:
    """Loads our trained Bunri-ConvNeXt-Tiny+GeM ONNX checkpoint and executes calibrated inference."""

    def __init__(self, root_dir: str | Path = ".") -> None:
        self.root_dir = Path(root_dir)
        self.model_path = self._resolve_model_path()
        self.metadata = self._load_metadata()

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        # Warmup pass for predictable first-request latency
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        self.session.run(None, {"input": dummy})

    def _resolve_model_path(self) -> Path:
        env_path = os.environ.get("BUNRI_ONNX_PATH")
        candidates = ([env_path] if env_path else []) + DEFAULT_WEIGHT_Candidates
        for rel in candidates:
            p = self.root_dir / rel if not Path(rel).is_absolute() else Path(rel)
            if p.exists():
                return p

        # Optional auto-download from GitHub Release if weight is not yet downloaded locally
        release_url = os.environ.get(
            "BUNRI_WEIGHT_URL",
            "https://github.com/qwertypotter6873/bunri-vision/releases/download/v1.0.0/bunri_convnext_tiny_int8.onnx",
        )
        fallback = self.root_dir / "models" / "bunri_convnext_tiny_int8.onnx"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        print(f"[MODEL] Local ONNX not found; downloading trained checkpoint from {release_url} ...")
        urllib.request.urlretrieve(release_url, fallback)
        return fallback

    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = self.root_dir / "models" / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "model_name": "Bunri-ConvNeXt-Tiny+GeM",
            "classes": CLASS_NAMES,
            "image_size": 224,
        }

    def predict(self, pil_img: Image.Image) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw_preview = np.asarray(
            pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        x_nchw, enhanced_224, quality_report = preprocess_for_inference(pil_img, image_size=224)

        calibrated_logits, cam_maps = self.session.run(None, {"input": x_nchw})
        logits_1d = calibrated_logits[0].astype(np.float64)
        logits_shifted = logits_1d - np.max(logits_1d)
        probs_1d = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted))

        top_idx = int(np.argmax(probs_1d))
        cam_overlay = _render_cam_overlay(enhanced_224, cam_maps[0, top_idx])
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        probs_dict = {CLASS_NAMES[i]: float(probs_1d[i]) for i in range(len(CLASS_NAMES))}
        decision = build_decision_advice(probs_dict)

        return {
            **decision,
            "inference_ms": elapsed_ms,
            "model_checkpoint": self.model_path.name,
            "model_metadata": self.metadata,
            "quality_report": asdict(quality_report),
            "visuals": {
                "raw_image_url": _encode_rgb_to_data_url(raw_preview),
                "preprocessed_image_url": _encode_rgb_to_data_url(enhanced_224),
                "cam_overlay_url": _encode_rgb_to_data_url(cam_overlay),
            },
        }
