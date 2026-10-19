"""Check the data composition before training: is each split usable and fair?

Reports, per split: patients, videos, frames, frames per patient (min/median/max),
label source mix, and class balance — for classification the label counts, for
segmentation the share of frames that contain each class and its mean area.
Flags problems that silently ruin a study:
  * a class missing from validation or test (its metric cannot be computed)
  * one patient contributing > 30% of a split's frames (score = that patient)
  * a minority class < 5% of training frames (consider class weighting / more cases)
  * labels from different sources mixed inside test (state it in Methods)

Usage:
  python data_audit.py --manifest split/split_manifest.csv --config config.json --out audit/
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sv_common import load_config, read_manifest, usable_rows, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    classes, task = cfg["class_names"], cfg["task_type"]
    base = Path(args.base) if args.base else Path(args.manifest).resolve().parent
    rows = read_manifest(args.manifest)
    rep, warns, lines = {"status_counts": dict(Counter(r.get("annotation_status", "") for r in rows))}, [], []
    lines.append("| Split | Patients | Videos | Frames | Frames/patient (min–median–max) | " +
                 ("Label counts" if task == "classification" else "Frames containing class (mean area %)") + " | Label source |")
    lines.append("|---|---:|---:|---:|---|---|---|")
    for s in ("train", "val", "test"):
        rs = usable_rows(rows, s)
        if not rs:
            continue
        per_p = Counter(r["patient_id"] for r in rs)
        fpp = sorted(per_p.values())
        d = {"patients": len(per_p), "videos": len({r["video_id"] for r in rs}), "frames": len(rs),
             "frames_per_patient": [fpp[0], int(np.median(fpp)), fpp[-1]],
             "label_source": dict(Counter(r.get("label_source", "unspecified") for r in rs))}
        top_p, top_n = per_p.most_common(1)[0]
        if len(per_p) >= 3 and top_n / len(rs) > 0.30:
            warns.append(f"{s}: patient {top_p} contributes {top_n / len(rs):.0%} of frames")
        if task == "classification":
            cnt = Counter(classes[int(r["label"])] for r in rs)
            d["label_counts"] = {c: cnt.get(c, 0) for c in classes}
            for c in classes:
                if s in ("val", "test") and cnt.get(c, 0) == 0:
                    warns.append(f"{s}: class '{c}' absent — its recall/F1 cannot be estimated")
                if s == "train" and cnt.get(c, 0) / len(rs) < 0.05:
                    warns.append(f"train: class '{c}' is {cnt.get(c, 0) / len(rs):.1%} of frames — minority class")
            cls_txt = ", ".join(f"{c}={n}" for c, n in d["label_counts"].items())
        else:
            present, area = Counter(), defaultdict(list)
            for r in rs:
                p = Path(r["mask_path"])
                m = np.asarray(Image.open(p if p.is_absolute() else base / p))
                valid = m != 255
                for k in range(1, len(classes)):
                    frac = float(((m == k) & valid).sum() / max(valid.sum(), 1))
                    if frac > 0:
                        present[classes[k]] += 1
                        area[classes[k]].append(frac)
            d["frames_with_class"] = {c: present.get(c, 0) for c in classes[1:]}
            d["mean_area_when_present"] = {c: round(float(np.mean(area[c])), 4) if area[c] else None for c in classes[1:]}
            for c in classes[1:]:
                if s in ("val", "test") and present.get(c, 0) == 0:
                    warns.append(f"{s}: class '{c}' never present — Dice for it cannot be estimated")
            cls_txt = ", ".join(f"{c}={present.get(c, 0)} ({100 * (d['mean_area_when_present'][c] or 0):.1f}%)" for c in classes[1:])
        if s == "test" and len(d["label_source"]) > 1:
            warns.append(f"test: labels come from several sources {d['label_source']} — report this in Methods")
        rep[s] = d
        src = ", ".join(f"{k}={v}" for k, v in d["label_source"].items())
        lines.append(f"| {s} | {d['patients']} | {d['videos']} | {d['frames']} | {fpp[0]}–{int(np.median(fpp))}–{fpp[-1]} | {cls_txt} | {src} |")
    rep["warnings"] = warns
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(rep, out / "data_audit.json")
    md = "\n".join(lines) + ("\n\n" + "\n".join(f"⚠ {w}" for w in warns) if warns else "\n\nNo composition warnings.")
    (out / "data_audit.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
