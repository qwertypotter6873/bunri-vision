"""Dataset curation, perceptual-hash deduplication, stratified splitting, and PyTorch Dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

from src.domain_rules import CLASS_NAMES
from src.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    aspect_preserving_center_crop_resize,
    compute_image_stats,
    preprocess_rgb_array,
)


def curate_and_split_dataset(
    raw_root: str | Path,
    output_manifest_csv: str | Path,
    stats_json_path: str | Path,
    phash_dedup_threshold: int = 4,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Scan RealWaste, prune near-duplicate frames (pHash dist <= threshold), and create 70/15/15 stratified split."""
    raw_root = Path(raw_root)
    records = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = raw_root / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing expected class folder: {cls_dir}")
        for fname in sorted(os.listdir(cls_dir)):
            fpath = cls_dir / fname
            if not fpath.is_file() or fname.startswith("."):
                continue
            im = Image.open(fpath).convert("RGB")
            ph = imagehash.phash(im, hash_size=8)
            rgb = np.asarray(im, dtype=np.uint8)
            lum_m, con_s, cast = compute_image_stats(rgb)
            records.append(
                {
                    "filepath": str(fpath),
                    "relpath": f"{cls_name}/{fname}",
                    "filename": fname,
                    "class_name": cls_name,
                    "label": cls_idx,
                    "width": im.width,
                    "height": im.height,
                    "phash_hex": str(ph),
                    "phash_bits": ph.hash.flatten().astype(np.uint8),
                    "raw_lum_mean": lum_m,
                    "raw_contrast_std": con_s,
                    "raw_color_cast": cast,
                }
            )

    df_all = pd.DataFrame(records)
    raw_total = len(df_all)
    raw_class_counts = df_all["class_name"].value_counts().reindex(CLASS_NAMES).to_dict()

    # Step 1 Preprocessing: Perceptual Hash near-duplicate pruning within & across classes (Hamming <= 4)
    bits = np.stack(df_all["phash_bits"].values)
    n = len(df_all)
    keep_mask = np.ones(n, dtype=bool)
    duplicate_pairs_before = 0
    removed_records = []

    for i in range(n):
        if i + 1 < n:
            dists = np.abs(bits[i + 1 :].astype(np.int16) - bits[i].astype(np.int16)).sum(axis=1)
            near_idxs = np.where(dists <= phash_dedup_threshold)[0] + (i + 1)
            duplicate_pairs_before += int(len(near_idxs))
            if keep_mask[i]:
                for j in near_idxs:
                    if keep_mask[j]:
                        keep_mask[j] = False
                        removed_records.append(
                            {
                                "kept": df_all.iloc[i]["relpath"],
                                "removed": df_all.iloc[j]["relpath"],
                                "hamming_dist": int(dists[j - (i + 1)]),
                            }
                        )

    df_clean = df_all[keep_mask].copy().reset_index(drop=True)

    # Verify zero remaining near-duplicate pairs <= threshold
    clean_bits = np.stack(df_clean["phash_bits"].values)
    remaining_near_pairs = 0
    for i in range(len(df_clean) - 1):
        dists = np.abs(clean_bits[i + 1 :].astype(np.int16) - clean_bits[i].astype(np.int16)).sum(axis=1)
        remaining_near_pairs += int((dists <= phash_dedup_threshold).sum())

    # Stratified 70% Train / 15% Val / 15% Test split with fixed seed=42
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(df_clean, df_clean["label"]))

    df_temp = df_clean.iloc[temp_idx].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_sub_idx, test_sub_idx = next(sss2.split(df_temp, df_temp["label"]))

    splits = np.array([""] * len(df_clean), dtype=object)
    splits[train_idx] = "train"
    splits[temp_idx[val_sub_idx]] = "val"
    splits[temp_idx[test_sub_idx]] = "test"
    df_clean["split"] = splits

    # Verify cross-split leakage pairs (train vs test) <= threshold
    tr_bits = np.stack(df_clean[df_clean["split"] == "train"]["phash_bits"].values)
    te_bits = np.stack(df_clean[df_clean["split"] == "test"]["phash_bits"].values)
    cross_leakage_pairs = 0
    for b in te_bits:
        dists = np.abs(tr_bits.astype(np.int16) - b.astype(np.int16)).sum(axis=1)
        cross_leakage_pairs += int((dists <= phash_dedup_threshold).sum())

    # Compute After-Preprocessing image statistics on clean dataset for quantitative Before/After report
    after_lum, after_con, after_cast = [], [], []
    for fpath in df_clean["filepath"]:
        rgb = np.asarray(Image.open(fpath).convert("RGB"), dtype=np.uint8)
        proc = preprocess_rgb_array(rgb, image_size=224, apply_domain_enhancement=True)
        lm, cs, ct = compute_image_stats(proc)
        after_lum.append(lm)
        after_con.append(cs)
        after_cast.append(ct)

    df_clean["after_lum_mean"] = after_lum
    df_clean["after_contrast_std"] = after_con
    df_clean["after_color_cast"] = after_cast

    export_df = df_clean.drop(columns=["phash_bits"])
    Path(output_manifest_csv).parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_manifest_csv, index=False)

    split_counts = (
        export_df.groupby(["class_name", "split"]).size().unstack(fill_value=0).reindex(CLASS_NAMES)
    )

    stats = {
        "seed": seed,
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "raw_total_images": int(raw_total),
        "deduplicated_total_images": int(len(export_df)),
        "removed_near_duplicates": int(raw_total - len(export_df)),
        "duplicate_pairs_before_dedup": int(duplicate_pairs_before),
        "duplicate_pairs_after_dedup": int(remaining_near_pairs),
        "train_test_cross_leakage_pairs": int(cross_leakage_pairs),
        "removed_examples": removed_records,
        "split_totals": export_df["split"].value_counts().to_dict(),
        "raw_class_counts": {k: int(v) for k, v in raw_class_counts.items()},
        "clean_class_split_counts": {
            cls: {sp: int(split_counts.loc[cls, sp]) for sp in ["train", "val", "test"]}
            for cls in CLASS_NAMES
        },
        "before_preprocessing_stats": {
            "luminance_mean": round(float(export_df["raw_lum_mean"].mean()), 3),
            "luminance_std": round(float(export_df["raw_lum_mean"].std()), 3),
            "contrast_std_mean": round(float(export_df["raw_contrast_std"].mean()), 3),
            "contrast_std_p10": round(float(np.percentile(export_df["raw_contrast_std"], 10)), 3),
            "color_cast_mean": round(float(export_df["raw_color_cast"].mean()), 4),
            "color_cast_p95": round(float(np.percentile(export_df["raw_color_cast"], 95)), 4),
        },
        "after_preprocessing_stats": {
            "luminance_mean": round(float(export_df["after_lum_mean"].mean()), 3),
            "luminance_std": round(float(export_df["after_lum_mean"].std()), 3),
            "contrast_std_mean": round(float(export_df["after_contrast_std"].mean()), 3),
            "contrast_std_p10": round(float(np.percentile(export_df["after_contrast_std"], 10)), 3),
            "color_cast_mean": round(float(export_df["after_color_cast"].mean()), 4),
            "color_cast_p95": round(float(np.percentile(export_df["after_color_cast"], 95)), 4),
        },
    }
    Path(stats_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return export_df, stats


class RealWasteDataset(Dataset):
    """Pre-cached in-memory RGB tensor dataset for fast multi-model GPU training."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        is_train: bool = True,
        apply_domain_enhancement: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.is_train = is_train
        self.apply_domain_enhancement = apply_domain_enhancement
        self.labels = self.df["label"].values.astype(np.int64)

        # Pre-process base 224x224 (or 256x256 for train random crop) in RAM (~550MB for 3300 imgs)
        base_size = 248 if is_train else image_size
        self.cached_images: List[Image.Image] = []
        for fpath in self.df["filepath"]:
            rgb = np.asarray(Image.open(fpath).convert("RGB"), dtype=np.uint8)
            if apply_domain_enhancement:
                proc = preprocess_rgb_array(
                    rgb,
                    image_size=base_size,
                    apply_domain_enhancement=True,
                    crop_ratio=0.92 if is_train else 0.90,
                )
            else:
                proc = cv2.resize(rgb, (base_size, base_size), interpolation=cv2.INTER_AREA)
            self.cached_images.append(Image.fromarray(proc))

        if is_train and apply_domain_enhancement:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.76, 1.0), ratio=(0.88, 1.14)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.25),
                    transforms.RandomRotation(degrees=20),
                    transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.15, hue=0.03),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
                    transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), value="random"),
                ]
            )
        elif is_train and not apply_domain_enhancement:
            # Naive baseline augmentation (only random crop + flip) for ablation comparison
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.cached_images[idx]
        tensor = self.transform(img)
        return tensor, int(self.labels[idx])


def compute_effective_number_weights(labels: np.ndarray, num_classes: int = 9, beta: float = 0.999) -> torch.Tensor:
    """Cui et al. (CVPR 2019) Class-Balanced Loss weights based on effective number of samples."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)
