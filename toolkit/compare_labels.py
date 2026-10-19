"""Agreement between two label sets on the frames both labelled.

Typical uses: participant vs reference labels, annotator 1 vs annotator 2,
first vs second read by the same annotator (intra-rater).

  segmentation   per-frame Dice per class, mean / median, worst frames
  classification percent agreement and Cohen's kappa

Usage:
  python compare_labels.py --a labeled_ref/labeled_manifest.csv --b labeled_me/labeled_manifest.csv \
      --config config.json --out agreement/ [--name-a reference --name-b me]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sv_common import (confusion, dice_iou, load_config, read_manifest, seg_case_stats,  # noqa: E402
                       write_json, write_manifest)


def kappa(a: list[int], b: list[int], k: int) -> float | None:
    m = confusion(a, b, k).astype(float)
    n = m.sum()
    if n == 0:
        return None
    po = np.trace(m) / n
    pe = float((m.sum(0) * m.sum(1)).sum()) / n ** 2
    return None if pe == 1 else float((po - pe) / (1 - pe))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--only-source-b", default="", help="use only B rows whose label_source equals this")
    args = ap.parse_args()
    cfg = load_config(args.config)
    classes, k = cfg["class_names"], len(cfg["class_names"])
    A = {r["sample_id"]: r for r in read_manifest(args.a) if r.get("annotation_status") == "done"}
    B = {r["sample_id"]: r for r in read_manifest(args.b) if r.get("annotation_status") == "done"
         and (not args.only_source_b or r.get("label_source") == args.only_source_b)}
    both = sorted(set(A) & set(B))
    if not both:
        sys.exit("[STOP] no frame is labelled in both sets")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {"n_frames_compared": len(both), "a": args.name_a, "b": args.name_b,
                         "task_type": cfg["task_type"]}
    if cfg["task_type"] == "segmentation":
        per = {c: [] for c in classes[1:]}
        for s in both:
            ma = np.asarray(Image.open(A[s]["mask_path"]))
            mb = np.asarray(Image.open(B[s]["mask_path"]).resize(ma.shape[::-1], Image.NEAREST))
            st = seg_case_stats(mb, ma, k)
            rec = {"sample_id": s, "patient_id": A[s]["patient_id"]}
            for c in range(1, k):
                d = dice_iou(st[c])[0]
                rec[f"dice_{classes[c]}"] = d
                if d is not None:
                    per[classes[c]].append(d)
            rows.append(rec)
        summary["per_class"] = {c: {"mean_dice": float(np.mean(v)) if v else None,
                                    "median_dice": float(np.median(v)) if v else None,
                                    "min_dice": float(np.min(v)) if v else None, "n": len(v)}
                                for c, v in per.items()}
        key = f"dice_{classes[1]}"
        summary["lowest_agreement_frames"] = [r["sample_id"] for r in sorted(
            rows, key=lambda r: (r[key] is None, r[key] if r[key] is not None else 1))[:5]]
    else:
        a = [int(A[s]["label"]) for s in both]
        b = [int(B[s]["label"]) for s in both]
        summary["percent_agreement"] = float(np.mean(np.array(a) == np.array(b)))
        summary["cohen_kappa"] = kappa(a, b, k)
        summary["confusion_a_rows_b_cols"] = confusion(a, b, k).tolist()
        rows = [{"sample_id": s, args.name_a: classes[x], args.name_b: classes[y]} for s, x, y in zip(both, a, b)]
    write_manifest(rows, out / "agreement_per_frame.csv", fields=list(rows[0]))
    write_json(summary, out / "agreement_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
