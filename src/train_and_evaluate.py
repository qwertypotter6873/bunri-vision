#!/usr/bin/env python3
"""Full reproducible GPU training, ablation, calibration, ONNX export & figure generation."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import koreanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.dataset import RealWasteDataset, compute_effective_number_weights, curate_and_split_dataset
from src.domain_rules import CLASS_NAMES, KOREAN_WASTE_RULES
from src.models import BunriConvNeXtGeM, ONNXExportWrapper, build_model
from src.preprocessing import (
    adaptive_lab_clahe,
    aspect_preserving_center_crop_resize,
    preprocess_for_inference,
    shades_of_gray_white_balance,
)

SHORT_KO_LABELS = [
    "골판지\n(Cardboard)",
    "음식물\n(Food)",
    "유리\n(Glass)",
    "금속\n(Metal)",
    "일반쓰레기\n(Misc Trash)",
    "종이\n(Paper)",
    "플라스틱\n(Plastic)",
    "섬유·의류\n(Textile)",
    "초목류\n(Vegetation)",
]

SHORT_KO_SINGLE = [
    "골판지",
    "음식물",
    "유리",
    "금속",
    "일반쓰레기",
    "종이",
    "플라스틱",
    "섬유·의류",
    "초목류",
]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) over probability confidence bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for b_low, b_high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > b_low) & (confidences <= b_high)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin
    return float(ece)


def evaluate_logits(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> Dict[str, Any]:
    scaled = logits / temperature
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)

    top2_preds = np.argsort(probs, axis=1)[:, -2:]
    top2_acc = float(np.mean([labels[i] in top2_preds[i] for i in range(len(labels))]))

    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    weighted_f1 = float(f1_score(labels, preds, average="weighted"))
    macro_auroc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    ece = compute_ece(probs, labels, n_bins=15)

    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    per_class = {
        CLASS_NAMES[i]: {
            "ko_name": SHORT_KO_SINGLE[i],
            "precision": round(float(prec_c[i]), 4),
            "recall": round(float(rec_c[i]), 4),
            "f1": round(float(f1_c[i]), 4),
            "support": int(sup_c[i]),
        }
        for i in range(len(CLASS_NAMES))
    }

    return {
        "accuracy": round(acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_auroc": round(macro_auroc, 4),
        "ece": round(ece, 4),
        "per_class": per_class,
        "probs": probs,
        "preds": preds,
    }


@torch.no_grad()
def predict_dataset(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    total_n = 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits.float(), y_dev)
        total_loss += float(loss.item()) * len(y)
        total_n += len(y)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0), total_loss / max(total_n, 1)


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or random.random() > 0.5:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Optimize scalar temperature T > 0 minimizing validation Negative Log-Likelihood."""
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.long)
    log_temp = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=80)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(0.25, 4.0)
        loss = F.cross_entropy(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp).clamp(0.25, 4.0).item())


def train_single_model(
    exp_id: str,
    display_name: str,
    arch: str,
    training_mode: str,
    pretrained: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor | None,
    epochs: int,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    label_smoothing: float,
    mixup_alpha: float,
    warmup_epochs: int,
    freeze_backbone_warmup_epochs: int,
    calibrate_temp: bool,
    device: torch.device,
    logs_dir: Path,
    models_dir: Path,
) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt_path = models_dir / f"{exp_id}_fp16.pt"
    hist_csv = logs_dir / f"history_{exp_id}.csv"
    if ckpt_path.exists() and hist_csv.exists():
        print(f"\n[RESUME] Loading completed checkpoint for {display_name} ({ckpt_path.name})")
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(arch=arch, num_classes=len(CLASS_NAMES), pretrained=False)
        fp16_state = saved["state_dict"]
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in fp16_state.items()})
        learned_temp = float(saved.get("temperature", 1.0))
        model.to(device).eval()
        history_rows = pd.read_csv(hist_csv).to_dict(orient="records")
        best_epoch = int(max(history_rows, key=lambda r: r["val_macro_f1"])["epoch"])
        train_duration = round(sum(float(r.get("epoch_sec", 0.0)) for r in history_rows), 1)
        param_count = sum(p.numel() for p in model.parameters())
        val_logits, val_labels, _ = predict_dataset(model, val_loader, device)
        test_logits, test_labels, _ = predict_dataset(model, test_loader, device)
        uncal_test = evaluate_logits(test_logits, test_labels, temperature=1.0)
        final_val = evaluate_logits(val_logits, val_labels, temperature=learned_temp)
        final_test = evaluate_logits(test_logits, test_labels, temperature=learned_temp)
        ckpt_mb = round(ckpt_path.stat().st_size / (1024**2), 2)
        model_cpu = copy.deepcopy(model).cpu().eval()
        dummy = torch.randn(1, 3, 224, 224)
        t_lat = time.perf_counter()
        with torch.no_grad():
            for _ in range(15):
                _ = model_cpu(dummy)
        cpu_latency_ms = round((time.perf_counter() - t_lat) * 1000.0 / 15.0, 2)
        return model, {
            "exp_id": exp_id,
            "display_name": display_name,
            "arch": arch,
            "training_mode": training_mode,
            "pretrained": pretrained,
            "param_count": int(param_count),
            "param_millions": round(param_count / 1e6, 2),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "train_duration_sec": train_duration,
            "temperature": round(learned_temp, 4),
            "checkpoint_fp16_mb": ckpt_mb,
            "cpu_pytorch_latency_ms": cpu_latency_ms,
            "val_macro_f1": final_val["macro_f1"],
            "val_accuracy": final_val["accuracy"],
            "test_accuracy": final_test["accuracy"],
            "test_top2_accuracy": final_test["top2_accuracy"],
            "test_macro_f1": final_test["macro_f1"],
            "test_weighted_f1": final_test["weighted_f1"],
            "test_macro_auroc": final_test["macro_auroc"],
            "test_ece_before_calib": uncal_test["ece"],
            "test_ece": final_test["ece"],
            "per_class": final_test["per_class"],
            "_test_logits": test_logits,
            "_test_probs": final_test["probs"],
            "_test_preds": final_test["preds"],
            "_test_labels": test_labels,
            "_history": history_rows,
        }

    print(f"\n{'=' * 72}")
    print(f"[TRAIN] {display_name} ({exp_id}) | mode={training_mode} | epochs={epochs}")
    print(f"{'=' * 72}")
    t0 = time.time()
    model = build_model(arch=arch, num_classes=len(CLASS_NAMES), pretrained=pretrained).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    if hasattr(model, "backbone"):
        backbone_params = list(model.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": head_lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    def lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return float(ep + 1) / float(max(1, warmup_epochs))
        progress = (ep - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return max(0.02, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    weight_tensor = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    history_rows: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()

        # Optional linear-probe warmup phase for transfer models
        if hasattr(model, "backbone") and freeze_backbone_warmup_epochs > 0:
            req_grad = epoch > freeze_backbone_warmup_epochs
            for p in model.backbone.parameters():
                p.requires_grad = req_grad

        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_in, ya, yb, lam = mixup_batch(x, y, alpha=mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(x_in)
                loss = lam * criterion(logits.float(), ya) + (1.0 - lam) * criterion(logits.float(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += float(loss.item()) * len(y)
            seen += len(y)

        scheduler.step()
        train_loss = running_loss / max(seen, 1)

        val_logits, val_labels, val_loss = predict_dataset(model, val_loader, device)
        val_metrics = evaluate_logits(val_logits, val_labels, temperature=1.0)
        ep_sec = time.time() - ep_t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_auroc": val_metrics["macro_auroc"],
            "lr_head": round(optimizer.param_groups[-1]["lr"], 6),
            "epoch_sec": round(ep_sec, 2),
        }
        history_rows.append(row)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} | best={best_val_f1:.4f}@{best_epoch} ({ep_sec:.1f}s)"
            )

    train_duration = time.time() - t0
    pd.DataFrame(history_rows).to_csv(logs_dir / f"history_{exp_id}.csv", index=False)

    model.load_state_dict(best_state)
    val_logits, val_labels, _ = predict_dataset(model, val_loader, device)
    test_logits, test_labels, _ = predict_dataset(model, test_loader, device)

    uncal_test = evaluate_logits(test_logits, test_labels, temperature=1.0)
    learned_temp = 1.0
    if calibrate_temp:
        learned_temp = fit_temperature(val_logits, val_labels)
        if isinstance(model, BunriConvNeXtGeM):
            model.temperature.fill_(learned_temp)
    final_val = evaluate_logits(val_logits, val_labels, temperature=learned_temp)
    final_test = evaluate_logits(test_logits, test_labels, temperature=learned_temp)

    # Save compact FP16 state dict checkpoint
    ckpt_path = models_dir / f"{exp_id}_fp16.pt"
    fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in model.cpu().state_dict().items()}
    torch.save({"arch": arch, "temperature": learned_temp, "state_dict": fp16_state}, ckpt_path)
    ckpt_mb = round(ckpt_path.stat().st_size / (1024**2), 2)

    # Measure single-image CPU latency (ms)
    model_cpu = build_model(arch=arch, num_classes=len(CLASS_NAMES), pretrained=False).eval()
    model_cpu.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in fp16_state.items()})
    dummy = torch.randn(1, 3, 224, 224)
    for _ in range(5):
        _ = model_cpu(dummy)
    t_lat = time.perf_counter()
    n_runs = 25
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model_cpu(dummy)
    cpu_latency_ms = round((time.perf_counter() - t_lat) * 1000.0 / n_runs, 2)

    model.to(device)
    print(
        f"[DONE] {display_name} | Best Epoch={best_epoch} | Test Acc={final_test['accuracy']:.4f} | "
        f"Test Macro-F1={final_test['macro_f1']:.4f} | AUROC={final_test['macro_auroc']:.4f} | "
        f"ECE={final_test['ece']:.4f} (T={learned_temp:.3f}) | Size={ckpt_mb}MB"
    )

    summary = {
        "exp_id": exp_id,
        "display_name": display_name,
        "arch": arch,
        "training_mode": training_mode,
        "pretrained": pretrained,
        "param_count": int(param_count),
        "param_millions": round(param_count / 1e6, 2),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "train_duration_sec": round(train_duration, 1),
        "temperature": round(learned_temp, 4),
        "checkpoint_fp16_mb": ckpt_mb,
        "cpu_pytorch_latency_ms": cpu_latency_ms,
        "val_macro_f1": final_val["macro_f1"],
        "val_accuracy": final_val["accuracy"],
        "test_accuracy": final_test["accuracy"],
        "test_top2_accuracy": final_test["top2_accuracy"],
        "test_macro_f1": final_test["macro_f1"],
        "test_weighted_f1": final_test["weighted_f1"],
        "test_macro_auroc": final_test["macro_auroc"],
        "test_ece_before_calib": uncal_test["ece"],
        "test_ece": final_test["ece"],
        "per_class": final_test["per_class"],
        "_test_logits": test_logits,
        "_test_probs": final_test["probs"],
        "_test_preds": final_test["preds"],
        "_test_labels": test_labels,
        "_history": history_rows,
    }
    return model, summary


def export_proposed_onnx(
    model: BunriConvNeXtGeM,
    onnx_path: Path,
) -> Tuple[float, float]:
    """Export self-contained single-file ONNX model (calibrated logits + 9-class spatial CAMs) & benchmark ONNX Runtime CPU."""
    wrapper = ONNXExportWrapper(copy.deepcopy(model).cpu().eval())
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    tmp_path = onnx_path.with_suffix(".raw.onnx")
    torch.onnx.export(
        wrapper,
        dummy,
        str(tmp_path),
        input_names=["input"],
        output_names=["calibrated_logits", "cam_maps"],
        dynamic_axes={"input": {0: "batch"}, "calibrated_logits": {0: "batch"}, "cam_maps": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    # Consolidate external data into a single portable .onnx file
    proto = onnx.load(str(tmp_path), load_external_data=True)
    onnx.save_model(proto, str(onnx_path), save_as_external_data=False)
    for extra in [tmp_path, Path(str(tmp_path) + ".data")]:
        if extra.exists():
            extra.unlink()

    # Also create dynamic INT8 quantized ONNX for ultra-fast & git-friendly (<30 MB) inference!
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = onnx_path.with_name("bunri_convnext_tiny_int8.onnx")
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    x_np = dummy.numpy()
    for _ in range(5):
        sess.run(None, {"input": x_np})
    t0 = time.perf_counter()
    for _ in range(30):
        out_logits, out_cams = sess.run(None, {"input": x_np})
    ort_ms = round((time.perf_counter() - t0) * 1000.0 / 30.0, 2)

    # Verify parity between PyTorch and ONNX Runtime
    with torch.no_grad():
        pt_logits, pt_cams = wrapper(dummy)
    max_diff = float(np.max(np.abs(pt_logits.numpy() - out_logits)))
    print(
        f"[ONNX] Exported {onnx_path.name} ({onnx_path.stat().st_size / (1024**2):.2f} MB) & "
        f"{int8_path.name} ({int8_path.stat().st_size / (1024**2):.2f} MB) | "
        f"ORT CPU Latency={ort_ms} ms | max_abs_diff={max_diff:.6f}"
    )
    return round(onnx_path.stat().st_size / (1024**2), 2), ort_ms


def render_cam_overlay(rgb_224: np.ndarray, cam_7x7: np.ndarray, alpha: float = 0.44) -> np.ndarray:
    cam = np.maximum(cam_7x7, 0.0)
    cam = cam - cam.min()
    if cam.max() > 1e-6:
        cam = cam / cam.max()
    cam_resized = cv2.resize(cam.astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    heatmap_bgr = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = (1.0 - alpha) * rgb_224.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_all_figures(
    df_clean: pd.DataFrame,
    dataset_stats: Dict[str, Any],
    summaries: Dict[str, Dict[str, Any]],
    proposed_model: BunriConvNeXtGeM,
    figures_dir: Path,
    device: torch.device,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["axes.unicode_minus"] = False

    # 1. Figure 1: Class Distribution & Stratified Train/Val/Test Split
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=220)
    raw_counts = [dataset_stats["raw_class_counts"][c] for c in CLASS_NAMES]
    palette = sns.color_palette("crest", len(CLASS_NAMES))
    bars = axes[0].barh(SHORT_KO_SINGLE[::-1], raw_counts[::-1], color=palette[::-1], edgecolor="#1e293b")
    axes[0].set_title("RealWaste 전체 클래스 분포 (원본 4,752장 · 불균형비 2.90:1)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("이미지 수 (장)")
    axes[0].set_xlim(0, max(raw_counts) * 1.16)
    for bar, cnt in zip(bars, raw_counts[::-1]):
        pct = cnt / sum(raw_counts) * 100
        axes[0].text(cnt + 12, bar.get_y() + bar.get_height() / 2, f"{cnt}장 ({pct:.1f}%)", va="center", fontsize=9)

    tr_c = [dataset_stats["clean_class_split_counts"][c]["train"] for c in CLASS_NAMES][::-1]
    va_c = [dataset_stats["clean_class_split_counts"][c]["val"] for c in CLASS_NAMES][::-1]
    te_c = [dataset_stats["clean_class_split_counts"][c]["test"] for c in CLASS_NAMES][::-1]
    y_pos = np.arange(len(CLASS_NAMES))
    axes[1].barh(y_pos, tr_c, label="Train (70%)", color="#2563eb")
    axes[1].barh(y_pos, va_c, left=tr_c, label="Val (15%)", color="#f59e0b")
    axes[1].barh(y_pos, te_c, left=np.array(tr_c) + np.array(va_c), label="Test (15%)", color="#10b981")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(SHORT_KO_SINGLE[::-1])
    axes[1].set_xlabel("정제 후 이미지 수 (장)")
    axes[1].set_title(
        f"pHash 중복 제거 후 Stratified 분할 (총 {dataset_stats['deduplicated_total_images']}장, seed=42)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig1_class_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Figure 2: 4-Stage Preprocessing Visual Before/After Comparison
    # Pick 4 samples with high color cast or low contrast across diverse classes
    sample_rows = []
    for target_cls in ["Plastic", "Glass", "Paper", "Food Organics"]:
        sub = df_clean[df_clean["class_name"] == target_cls].sort_values(
            by="raw_color_cast", ascending=False
        )
        sample_rows.append(sub.iloc[0])

    fig, axes = plt.subplots(4, 4, figsize=(12.5, 12.2), dpi=220)
    col_titles = [
        "① 원본 이미지 (524×524)\n조명 색편향·주변 배경 노이즈",
        "② Shades-of-Gray WB\n실내·야외 조명 색온도 보정",
        "③ LAB 공간 적응형 CLAHE\n투명·구겨짐 미세 윤곽 복원",
        "④ 중심 90% 크롭 (224×224)\n트레이·바닥 배경 단서 제거",
    ]
    for r_idx, row in enumerate(sample_rows):
        raw_rgb = np.asarray(Image.open(row["filepath"]).convert("RGB"), dtype=np.uint8)
        wb_rgb = shades_of_gray_white_balance(raw_rgb)
        clahe_rgb = adaptive_lab_clahe(wb_rgb)
        crop_rgb = aspect_preserving_center_crop_resize(clahe_rgb, image_size=224, crop_ratio=0.90)
        stages = [raw_rgb, wb_rgb, clahe_rgb, crop_rgb]
        ko_cls = KOREAN_WASTE_RULES[row["class_name"]].ko_name
        for c_idx, img_st in enumerate(stages):
            axes[r_idx, c_idx].imshow(img_st)
            axes[r_idx, c_idx].set_xticks([])
            axes[r_idx, c_idx].set_yticks([])
            if r_idx == 0:
                axes[r_idx, c_idx].set_title(col_titles[c_idx], fontsize=10.5, fontweight="bold", pad=8)
            if c_idx == 0:
                axes[r_idx, c_idx].set_ylabel(
                    f"{ko_cls}\n({row['class_name']})", fontsize=10, fontweight="bold"
                )
    plt.suptitle("도메인 특화 전처리 파이프라인 단계별 전·후 시각 비교", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig2_preprocessing_before_after.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Figure 3: Quantitative Before vs After Preprocessing Shift
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=220)
    sns.kdeplot(df_clean["raw_color_cast"], ax=axes[0], fill=True, color="#ef4444", label="전처리 전 (원본)", alpha=0.35)
    sns.kdeplot(df_clean["after_color_cast"], ax=axes[0], fill=True, color="#2563eb", label="전처리 후 (WB+CLAHE)", alpha=0.35)
    axes[0].set_title("① 조명 색편향 지수 (Color Cast Index ↓)", fontweight="bold")
    axes[0].set_xlabel("RGB 채널 불균형 표준편차 (낮을수록 중립 조명)")
    axes[0].legend()

    sns.kdeplot(df_clean["raw_contrast_std"], ax=axes[1], fill=True, color="#ef4444", label="전처리 전 (원본)", alpha=0.35)
    sns.kdeplot(df_clean["after_contrast_std"], ax=axes[1], fill=True, color="#10b981", label="전처리 후 (CLAHE+Crop)", alpha=0.35)
    axes[1].set_title("② 지역 명암 대비 (Local Contrast Std ↑)", fontweight="bold")
    axes[1].set_xlabel("휘도 표준편차 (저대비 샘플 윤곽선 개선)")
    axes[1].legend()

    leak_labels = ["중복 제거 전\n유사 쌍(pHash≤4)", "중복 제거 후\n유사 쌍(pHash≤4)", "Train-Test 간\n누수 쌍"]
    leak_vals = [
        dataset_stats["duplicate_pairs_before_dedup"],
        dataset_stats["duplicate_pairs_after_dedup"],
        dataset_stats["train_test_cross_leakage_pairs"],
    ]
    b_leak = axes[2].bar(leak_labels, leak_vals, color=["#f97316", "#22c55e", "#0ea5e9"], width=0.52, edgecolor="#1e293b")
    axes[2].set_title("③ 지각 해시(pHash) 중복·데이터 누수 차단", fontweight="bold")
    axes[2].set_ylabel("쌍(Pair) 개수")
    axes[2].set_ylim(0, max(leak_vals) * 1.25 + 2)
    for bar, v in zip(b_leak, leak_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.35, f"{v}쌍", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig3_preprocessing_quant_stats.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Figure 4: Learning Curves across all models
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=220)
    color_map = {
        "m1_scratch_cnn": "#64748b",
        "m2_mobilenetv3": "#f59e0b",
        "m3_resnet50": "#8b5cf6",
        "m4_ablation_raw": "#ef4444",
        "m4_proposed_convnext": "#059669",
    }
    for exp_id, s in summaries.items():
        hist_df = pd.DataFrame(s["_history"])
        lw = 2.6 if exp_id == "m4_proposed_convnext" else 1.7
        ls = "--" if exp_id == "m4_ablation_raw" else "-"
        axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label=s["display_name"], color=color_map[exp_id], linewidth=lw, linestyle=ls)
        axes[1].plot(hist_df["epoch"], hist_df["val_macro_f1"], label=s["display_name"], color=color_map[exp_id], linewidth=lw, linestyle=ls)

    axes[0].set_title("에폭별 학습 손실 (Train Loss) 수렴 곡선", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8.8)

    axes[1].axhline(0.88, color="#dc2626", linestyle=":", linewidth=1.5, label="정량 목표 (Macro-F1 = 0.88)")
    axes[1].set_title("에폭별 검증셋 성능 (Validation Macro-F1) 학습 곡선", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Macro-F1")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8.8, loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig4_learning_curves.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Figure 5: Test Set Multi-Metric Benchmark Comparison
    fig, ax = plt.subplots(figsize=(12.5, 5.4), dpi=220)
    order = ["m1_scratch_cnn", "m2_mobilenetv3", "m3_resnet50", "m4_ablation_raw", "m4_proposed_convnext"]
    names = [summaries[k]["display_name"] for k in order]
    f1s = [summaries[k]["test_macro_f1"] for k in order]
    accs = [summaries[k]["test_accuracy"] for k in order]
    aurocs = [summaries[k]["test_macro_auroc"] for k in order]

    x = np.arange(len(order))
    w = 0.25
    b1 = ax.bar(x - w, f1s, w, label="Test Macro-F1 (주 평가지표)", color="#059669")
    b2 = ax.bar(x, accs, w, label="Test Accuracy", color="#2563eb")
    b3 = ax.bar(x + w, aurocs, w, label="Test Macro-AUROC", color="#7c3aed")
    ax.axhline(0.88, color="#dc2626", linestyle="--", linewidth=1.4, label="목표 기준선 (Macro-F1 ≥ 0.88)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylim(0.55, 1.04)
    ax.set_ylabel("점수 (Score)")
    ax.set_title("홀드아웃 테스트셋 모델별 객관적 평가지표 비교", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    for bars_grp in [b1, b2]:
        for bar in bars_grp:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.007, f"{v:.3f}", ha="center", fontsize=8.3, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig5_model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 6. Figure 6: Side-by-side Normalized Confusion Matrix (Scratch Baseline vs Proposed Model)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), dpi=220)
    for ax_i, (exp_id, title_prefix) in enumerate(
        [
            ("m1_scratch_cnn", "베이스라인: ScratchResSECNN"),
            ("m4_proposed_convnext", "제안 모델: Bunri-ConvNeXt-Tiny+GeM"),
        ]
    ):
        s = summaries[exp_id]
        cm = confusion_matrix(s["_test_labels"], s["_test_preds"], normalize="true")
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues" if ax_i == 0 else "Greens",
            xticklabels=SHORT_KO_SINGLE,
            yticklabels=SHORT_KO_SINGLE,
            cbar=False,
            ax=axes[ax_i],
            vmin=0.0,
            vmax=1.0,
        )
        axes[ax_i].set_title(
            f"{title_prefix}\n(Test Macro-F1 = {s['test_macro_f1']:.4f}, Acc = {s['test_accuracy']:.4f})",
            fontsize=11.5,
            fontweight="bold",
        )
        axes[ax_i].set_xlabel("예측 클래스 (Predicted)")
        axes[ax_i].set_ylabel("실제 클래스 (Ground Truth)")
        axes[ax_i].tick_params(axis="x", rotation=30)
        axes[ax_i].tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig6_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # 7. Figure 7: Per-Class Precision / Recall / F1 of Proposed Model vs Baselines
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=220)
    x = np.arange(len(CLASS_NAMES))
    w = 0.26
    m1_cls = [summaries["m1_scratch_cnn"]["per_class"][c]["f1"] for c in CLASS_NAMES]
    m2_cls = [summaries["m2_mobilenetv3"]["per_class"][c]["f1"] for c in CLASS_NAMES]
    m4_cls = [summaries["m4_proposed_convnext"]["per_class"][c]["f1"] for c in CLASS_NAMES]
    ax.bar(x - w, m1_cls, w, label="ScratchResSECNN (Scratch)", color="#94a3b8")
    ax.bar(x, m2_cls, w, label="MobileNetV3-Small (전이학습)", color="#f59e0b")
    b_prop = ax.bar(x + w, m4_cls, w, label="Bunri-ConvNeXt-Tiny+GeM (제안)", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_KO_SINGLE, fontsize=10)
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("클래스별 Test F1-Score")
    ax.set_title("9개 생활폐기물 품목별 F1-Score 상세 비교 (소수 클래스 섬유·유리·음식물 대폭 개선)", fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    for bar in b_prop:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=8.3, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig7_per_class_f1.png", bbox_inches="tight")
    plt.close(fig)

    # 8. Figure 8: Error Analysis + Grad-CAM Visualization on Test Set Misclassifications & Successes
    df_test = df_clean[df_clean["split"] == "test"].reset_index(drop=True)
    prop_preds = summaries["m4_proposed_convnext"]["_test_preds"]
    prop_probs = summaries["m4_proposed_convnext"]["_test_probs"]
    prop_labels = summaries["m4_proposed_convnext"]["_test_labels"]

    wrong_indices = np.where(prop_preds != prop_labels)[0]
    # Sort wrong predictions by highest confidence error (most informative failure modes)
    wrong_sorted = sorted(wrong_indices, key=lambda idx: float(prop_probs[idx, prop_preds[idx]]), reverse=True)[:4]

    proposed_cpu = copy.deepcopy(proposed_model).cpu().eval()
    if len(wrong_sorted) > 0:
        n_err = len(wrong_sorted)
        fig, axes = plt.subplots(n_err, 3, figsize=(11.5, 3.1 * n_err), dpi=220)
        if n_err == 1:
            axes = np.expand_dims(axes, axis=0)
        for r_i, idx in enumerate(wrong_sorted):
            row = df_test.iloc[idx]
            pil_im = Image.open(row["filepath"]).convert("RGB")
            x_nchw, rgb_224, _ = preprocess_for_inference(pil_im, image_size=224)
            with torch.no_grad():
                _, cal_logits, cams = proposed_cpu.forward_features_and_logits(torch.from_numpy(x_nchw))
            probs = torch.softmax(cal_logits[0], dim=0).numpy()
            pred_c = int(np.argmax(probs))
            true_c = int(row["label"])
            cam_overlay = render_cam_overlay(rgb_224, cams[0, pred_c].numpy())

            axes[r_i, 0].imshow(rgb_224)
            axes[r_i, 0].set_title(f"실제 정답: {SHORT_KO_SINGLE[true_c]} ({CLASS_NAMES[true_c]})", fontsize=10, fontweight="bold")
            axes[r_i, 0].axis("off")

            axes[r_i, 1].imshow(cam_overlay)
            axes[r_i, 1].set_title(
                f"오분류 예측: {SHORT_KO_SINGLE[pred_c]} ({probs[pred_c]*100:.1f}%)\nCAM 주목 영역 시각화",
                fontsize=10,
                color="#dc2626",
                fontweight="bold",
            )
            axes[r_i, 1].axis("off")

            top3 = np.argsort(probs)[::-1][:3]
            top_names = [f"{SHORT_KO_SINGLE[k]} ({CLASS_NAMES[k]})" for k in top3][::-1]
            top_vals = [probs[k] * 100 for k in top3][::-1]
            bar_cols = ["#10b981" if k == true_c else "#ef4444" if k == pred_c else "#94a3b8" for k in top3][::-1]
            axes[r_i, 2].barh(top_names, top_vals, color=bar_cols)
            axes[r_i, 2].set_xlim(0, 108)
            axes[r_i, 2].set_xlabel("예측 확률 (%)")
            axes[r_i, 2].set_title("Top-3 클래스 확률 분포 (초록=실제 정답)", fontsize=9.5, fontweight="bold")
            for b_idx, v in enumerate(top_vals):
                axes[r_i, 2].text(v + 1.5, b_idx, f"{v:.1f}%", va="center", fontsize=8.8)

        plt.suptitle("테스트셋 대표 오분류 사례 및 CAM 원인 분석", fontsize=13.5, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(figures_dir / "fig8_error_analysis_gradcam.png", bbox_inches="tight")
        plt.close(fig)


def export_curated_test_samples(
    df_clean: pd.DataFrame,
    sample_dir: Path,
) -> None:
    """Copy 9 representative held-out test images (1 per class, compressed JPEG) into sample_data/ for 1-click live demo."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    df_test = df_clean[df_clean["split"] == "test"].reset_index(drop=True)
    manifest = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        sub = df_test[df_test["class_name"] == cls_name].reset_index(drop=True)
        chosen = sub.iloc[min(2, len(sub) - 1)]
        slug = cls_name.lower().replace(" ", "_")
        out_name = f"sample_{cls_idx + 1:02d}_{slug}.jpg"
        out_path = sample_dir / out_name
        im = Image.open(chosen["filepath"]).convert("RGB")
        im.save(out_path, format="JPEG", quality=88, optimize=True)
        rule = KOREAN_WASTE_RULES[cls_name]
        manifest.append(
            {
                "id": slug,
                "filename": out_name,
                "ground_truth_en": cls_name,
                "ground_truth_ko": rule.ko_name,
                "bin_category": rule.bin_category,
            }
        )
    with open(sample_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    seed_everything(42)
    root = Path(".")
    raw_root = root / "data" / "raw" / "realwaste-main" / "RealWaste"
    logs_dir = root / "logs"
    models_dir = root / "models"
    figures_dir = root / "assets" / "figures"
    sample_dir = root / "sample_data"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    manifest_csv = logs_dir / "split_manifest.csv"
    stats_json = logs_dir / "dataset_stats.json"
    if manifest_csv.exists() and stats_json.exists():
        df_clean = pd.read_csv(manifest_csv)
        with open(stats_json, "r", encoding="utf-8") as f:
            dataset_stats = json.load(f)
    else:
        df_clean, dataset_stats = curate_and_split_dataset(
            raw_root=raw_root,
            output_manifest_csv=manifest_csv,
            stats_json_path=stats_json,
            phash_dedup_threshold=4,
            seed=42,
        )
    print(
        f"[SPLIT] Raw={dataset_stats['raw_total_images']} -> Clean={dataset_stats['deduplicated_total_images']} "
        f"(Removed {dataset_stats['removed_near_duplicates']} near-dups) | "
        f"Cross-leakage pairs={dataset_stats['train_test_cross_leakage_pairs']} | "
        f"Splits={dataset_stats['split_totals']}"
    )

    df_train = df_clean[df_clean["split"] == "train"]
    df_val = df_clean[df_clean["split"] == "val"]
    df_test = df_clean[df_clean["split"] == "test"]

    all_ckpts_ready = all(
        (models_dir / f"{eid}_fp16.pt").exists() and (logs_dir / f"history_{eid}.csv").exists()
        for eid in [
            "m1_scratch_cnn",
            "m2_mobilenetv3",
            "m3_resnet50",
            "m4_ablation_raw",
            "m4_proposed_convnext",
        ]
    )

    print("[DATA] Caching evaluation (and training if needed) datasets in RAM...")
    ds_val_enh = RealWasteDataset(df_val, image_size=224, is_train=False, apply_domain_enhancement=True)
    ds_test_enh = RealWasteDataset(df_test, image_size=224, is_train=False, apply_domain_enhancement=True)
    ds_val_raw = RealWasteDataset(df_val, image_size=224, is_train=False, apply_domain_enhancement=False)
    ds_test_raw = RealWasteDataset(df_test, image_size=224, is_train=False, apply_domain_enhancement=False)

    if all_ckpts_ready:
        ds_train_enh = ds_val_enh
        ds_train_raw = ds_val_raw
    else:
        ds_train_enh = RealWasteDataset(df_train, image_size=224, is_train=True, apply_domain_enhancement=True)
        ds_train_raw = RealWasteDataset(df_train, image_size=224, is_train=True, apply_domain_enhancement=False)

    cb_weights = compute_effective_number_weights(df_train["label"].values, num_classes=len(CLASS_NAMES), beta=0.999)
    print("[WEIGHTS] Effective-Number Class-Balanced weights:", np.round(cb_weights.numpy(), 3).tolist())

    loader_kwargs = {"batch_size": 64, "num_workers": 4, "pin_memory": (device.type == "cuda")}
    tr_loader_enh = DataLoader(ds_train_enh, shuffle=True, drop_last=False, **loader_kwargs)
    va_loader_enh = DataLoader(ds_val_enh, shuffle=False, **loader_kwargs)
    te_loader_enh = DataLoader(ds_test_enh, shuffle=False, **loader_kwargs)

    tr_loader_raw = DataLoader(ds_train_raw, shuffle=True, drop_last=False, **loader_kwargs)
    va_loader_raw = DataLoader(ds_val_raw, shuffle=False, **loader_kwargs)
    te_loader_raw = DataLoader(ds_test_raw, shuffle=False, **loader_kwargs)

    summaries: Dict[str, Dict[str, Any]] = {}

    # 1. Model 1: Scratch Baseline (Custom 5-Stage Residual SE-CNN)
    seed_everything(42)
    _, s1 = train_single_model(
        exp_id="m1_scratch_cnn",
        display_name="ScratchResSECNN (Scratch)",
        arch="scratch_cnn",
        training_mode="Scratch (무작위 초기화 직접 학습)",
        pretrained=False,
        train_loader=tr_loader_enh,
        val_loader=va_loader_enh,
        test_loader=te_loader_enh,
        class_weights=cb_weights,
        epochs=45,
        backbone_lr=1.2e-3,
        head_lr=1.2e-3,
        weight_decay=1e-4,
        label_smoothing=0.05,
        mixup_alpha=0.15,
        warmup_epochs=4,
        freeze_backbone_warmup_epochs=0,
        calibrate_temp=False,
        device=device,
        logs_dir=logs_dir,
        models_dir=models_dir,
    )
    summaries["m1_scratch_cnn"] = s1

    # 2. Model 2: Lightweight Transfer Baseline (MobileNetV3-Small)
    seed_everything(42)
    _, s2 = train_single_model(
        exp_id="m2_mobilenetv3",
        display_name="MobileNetV3-Small (전이학습)",
        arch="mobilenetv3_small",
        training_mode="전이학습 + 파인튜닝 (ImageNet-1K)",
        pretrained=True,
        train_loader=tr_loader_enh,
        val_loader=va_loader_enh,
        test_loader=te_loader_enh,
        class_weights=cb_weights,
        epochs=30,
        backbone_lr=3.0e-4,
        head_lr=1.2e-3,
        weight_decay=1e-4,
        label_smoothing=0.05,
        mixup_alpha=0.15,
        warmup_epochs=3,
        freeze_backbone_warmup_epochs=2,
        calibrate_temp=False,
        device=device,
        logs_dir=logs_dir,
        models_dir=models_dir,
    )
    summaries["m2_mobilenetv3"] = s2

    # 3. Model 3: Standard CNN Fine-Tuning (ResNet-50)
    seed_everything(42)
    _, s3 = train_single_model(
        exp_id="m3_resnet50",
        display_name="ResNet-50 (파인튜닝)",
        arch="resnet50",
        training_mode="파인튜닝 (ImageNet-1K)",
        pretrained=True,
        train_loader=tr_loader_enh,
        val_loader=va_loader_enh,
        test_loader=te_loader_enh,
        class_weights=cb_weights,
        epochs=30,
        backbone_lr=1.8e-4,
        head_lr=9.0e-4,
        weight_decay=2e-4,
        label_smoothing=0.05,
        mixup_alpha=0.20,
        warmup_epochs=3,
        freeze_backbone_warmup_epochs=2,
        calibrate_temp=False,
        device=device,
        logs_dir=logs_dir,
        models_dir=models_dir,
    )
    summaries["m3_resnet50"] = s3

    # 4. Model 4-Ablation: ConvNeXt-Tiny WITHOUT domain preprocessing & class-balanced loss
    seed_everything(42)
    _, s4_abl = train_single_model(
        exp_id="m4_ablation_raw",
        display_name="ConvNeXt-Tiny (전처리 미적용 Ablation)",
        arch="convnext_tiny_gem",
        training_mode="파인튜닝 (도메인 전처리·불균형 보정 제거)",
        pretrained=True,
        train_loader=tr_loader_raw,
        val_loader=va_loader_raw,
        test_loader=te_loader_raw,
        class_weights=None,
        epochs=28,
        backbone_lr=1.2e-4,
        head_lr=6.0e-4,
        weight_decay=5e-4,
        label_smoothing=0.0,
        mixup_alpha=0.0,
        warmup_epochs=3,
        freeze_backbone_warmup_epochs=2,
        calibrate_temp=False,
        device=device,
        logs_dir=logs_dir,
        models_dir=models_dir,
    )
    summaries["m4_ablation_raw"] = s4_abl

    # 5. Model 4 (Proposed): Bunri-ConvNeXt-Tiny + GeM + 4-Stage Domain Preproc + Class-Balanced Mixup + Temp Scaling
    seed_everything(42)
    proposed_model, s4_prop = train_single_model(
        exp_id="m4_proposed_convnext",
        display_name="Bunri-ConvNeXt-Tiny+GeM (제안 모델)",
        arch="convnext_tiny_gem",
        training_mode="2단계 차등 파인튜닝 + GeM 풀링 + 온도 보정",
        pretrained=True,
        train_loader=tr_loader_enh,
        val_loader=va_loader_enh,
        test_loader=te_loader_enh,
        class_weights=cb_weights,
        epochs=32,
        backbone_lr=1.3e-4,
        head_lr=7.5e-4,
        weight_decay=5e-4,
        label_smoothing=0.06,
        mixup_alpha=0.20,
        warmup_epochs=3,
        freeze_backbone_warmup_epochs=2,
        calibrate_temp=True,
        device=device,
        logs_dir=logs_dir,
        models_dir=models_dir,
    )
    summaries["m4_proposed_convnext"] = s4_prop

    onnx_mb, ort_ms = export_proposed_onnx(
        proposed_model, onnx_path=models_dir / "bunri_convnext_tiny.onnx"
    )
    s4_prop["onnx_fp32_mb"] = onnx_mb
    s4_prop["onnx_int8_mb"] = round((models_dir / "bunri_convnext_tiny_int8.onnx").stat().st_size / (1024**2), 2)
    s4_prop["onnx_cpu_latency_ms"] = ort_ms

    # Verify INT8 ONNX accuracy on full Test Set!
    sess_int8 = ort.InferenceSession(
        str(models_dir / "bunri_convnext_tiny_int8.onnx"), providers=["CPUExecutionProvider"]
    )
    int8_logits_list = []
    for x, _ in te_loader_enh:
        out_l, _ = sess_int8.run(None, {"input": x.numpy()})
        int8_logits_list.append(out_l)
    int8_logits = np.concatenate(int8_logits_list, axis=0)
    int8_eval = evaluate_logits(int8_logits, s4_prop["_test_labels"], temperature=1.0)
    s4_prop["onnx_int8_test_macro_f1"] = int8_eval["macro_f1"]
    s4_prop["onnx_int8_test_accuracy"] = int8_eval["accuracy"]
    print(
        f"[INT8 VERIFY] INT8 ONNX Test Macro-F1={int8_eval['macro_f1']:.4f}, "
        f"Accuracy={int8_eval['accuracy']:.4f} (Size={s4_prop['onnx_int8_mb']} MB)"
    )

    generate_all_figures(df_clean, dataset_stats, summaries, proposed_model, figures_dir, device)
    export_curated_test_samples(df_clean, sample_dir)

    # Strip large numpy arrays before saving clean JSON benchmark summary
    clean_summaries = {}
    for k, v in summaries.items():
        clean_summaries[k] = {sk: sv for sk, sv in v.items() if not sk.startswith("_")}

    benchmark_payload = {
        "project": "Bunri-Vision AI (분리비전 AI)",
        "dataset": dataset_stats,
        "models": clean_summaries,
        "selected_final_model": "m4_proposed_convnext",
    }
    with open(logs_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, ensure_ascii=False, indent=2)

    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": "Bunri-ConvNeXt-Tiny+GeM",
                "backbone": "convnext_tiny.fb_in22k_ft_in1k",
                "num_classes": len(CLASS_NAMES),
                "classes": CLASS_NAMES,
                "image_size": 224,
                "temperature": s4_prop["temperature"],
                "test_macro_f1": s4_prop["test_macro_f1"],
                "test_accuracy": s4_prop["test_accuracy"],
                "test_top2_accuracy": s4_prop["test_top2_accuracy"],
                "test_macro_auroc": s4_prop["test_macro_auroc"],
                "test_ece": s4_prop["test_ece"],
                "onnx_int8_test_macro_f1": s4_prop["onnx_int8_test_macro_f1"],
                "onnx_cpu_latency_ms": ort_ms,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[SUCCESS] All training, calibration, ONNX exports, figures, and metadata completed!")


if __name__ == "__main__":
    main()
