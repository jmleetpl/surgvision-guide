"""SurgVision toolkit — shared helpers (manifest, config, metrics, bootstrap).

Teaching/research scaffold. Not a clinical device. Every number this toolkit
prints comes from the files it was given; nothing is filled in from examples.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np

IGNORE_INDEX = 255
MANIFEST_FIELDS = [
    "sample_id", "patient_id", "video_id", "frame_idx", "time_sec",
    "image_path", "mask_path", "label", "split", "annotation_status",
]


# ---------------------------------------------------------------- config / io
def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    task = cfg.get("task_type")
    if task not in {"segmentation", "classification"}:
        raise ValueError(f"config.task_type must be segmentation|classification, got {task!r}")
    if not cfg.get("class_names"):
        raise ValueError("config.class_names is required (list of label names)")
    return cfg


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict], path: str | Path, fields: list[str] | None = None) -> None:
    fields = fields or MANIFEST_FIELDS
    extra = [k for r in rows for k in r if k not in fields]
    fields = fields + sorted(set(extra), key=extra.index)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def usable_rows(rows: list[dict], split: str | None = None) -> list[dict]:
    """Rows that are actually annotated. Unannotated frames are never negatives."""
    out = [r for r in rows if r.get("annotation_status", "done") == "done"]
    return [r for r in out if split is None or r.get("split") == split]


# ---------------------------------------------------------------- seg metrics
def seg_case_stats(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> list[dict]:
    """Per-class pixel counts for one image; ignore pixels (255) excluded."""
    valid = gt != IGNORE_INDEX
    stats = []
    for c in range(n_classes):
        p = (pred == c) & valid
        g = (gt == c) & valid
        stats.append({"tp": int((p & g).sum()), "fp": int((p & ~g).sum()),
                      "fn": int((~p & g).sum()), "gt_px": int(g.sum()), "pred_px": int(p.sum())})
    return stats


def dice_iou(s: dict) -> tuple[float | None, float | None]:
    """Dice/IoU for one class in one image. Both empty -> None (reported separately)."""
    denom = 2 * s["tp"] + s["fp"] + s["fn"]
    if denom == 0:
        return None, None
    return 2 * s["tp"] / denom, s["tp"] / (s["tp"] + s["fp"] + s["fn"])


# ---------------------------------------------------------------- cls metrics
def confusion(y_true: list[int], y_pred: list[int], k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def cls_metrics(y_true: list[int], y_pred: list[int], k: int) -> dict:
    m = confusion(y_true, y_pred, k)
    n = m.sum()
    recalls, precisions, f1s = [], [], []
    for c in range(k):
        tp, fn, fp = m[c, c], m[c].sum() - m[c, c], m[:, c].sum() - m[c, c]
        rec = tp / (tp + fn) if tp + fn else None
        pre = tp / (tp + fp) if tp + fp else None
        f1 = (2 * pre * rec / (pre + rec)) if (pre and rec) else (0.0 if rec is not None else None)
        recalls.append(rec), precisions.append(pre), f1s.append(f1)
    present = [r for r in recalls if r is not None]
    f1p = [f for f, r in zip(f1s, recalls) if r is not None and f is not None]
    return {
        "n": int(n),
        "accuracy": float(np.trace(m) / n) if n else None,
        "balanced_accuracy": float(np.mean(present)) if present else None,
        "macro_f1": float(np.mean(f1p)) if f1p else None,
        "per_class_recall": recalls, "per_class_precision": precisions, "per_class_f1": f1s,
        "confusion_matrix": m.tolist(),
    }


def auroc(y_true: list[int], score: list[float]) -> float | None:
    """Rank-based (Mann-Whitney) AUROC for one binary target; ties get half credit."""
    pos = [s for t, s in zip(y_true, score) if t == 1]
    neg = [s for t, s in zip(y_true, score) if t == 0]
    if not pos or not neg:
        return None
    s = np.array(score, float)
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                       # average ranks for ties
        idx = s == v
        ranks[idx] = ranks[idx].mean()
    r_pos = ranks[np.array(y_true) == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- bootstrap
def cluster_bootstrap(values_by_cluster: dict[str, list[float]], n_boot: int = 2000,
                      seed: int = 0, stat=np.mean) -> dict:
    """95% CI resampling whole clusters (patients/videos), not frames.

    Frames from one surgery are correlated; resampling frames would make the
    interval far too narrow. With < 5 clusters the CI is flagged unreliable.
    """
    clusters = [k for k, v in values_by_cluster.items() if v]
    if not clusters:
        return {"estimate": None, "ci95": [None, None], "n_clusters": 0, "reliable": False}
    point = float(stat([x for k in clusters for x in values_by_cluster[k]]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        boots.append(stat([x for k in pick for x in values_by_cluster[k]]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"estimate": point, "ci95": [float(lo), float(hi)], "n_clusters": len(clusters),
            "reliable": len(clusters) >= 5, "method": f"cluster bootstrap, {n_boot} resamples"}
