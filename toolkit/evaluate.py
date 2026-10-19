"""Evaluate a FROZEN checkpoint on another labelled set — external validation.

External = patients from another hospital, another period, another device, or a
public dataset, that played no part in training, model selection or threshold
choice. Nothing is re-trained or tuned here. The same metrics, bootstrap and
failure examples as the internal test are produced, plus a side-by-side table.

Usage:
  python evaluate.py --checkpoint runs/run01/checkpoint_best.pt --manifest ext/labeled_manifest.csv \
      --name external_hospitalB --out runs/run01/external_B [--internal runs/run01/metrics.json]
  (all rows with annotation_status=done are used; --split test restricts to one split)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model  # noqa: E402
from sv_common import read_manifest, usable_rows, write_json  # noqa: E402
from train import FrameSet, pick_device, predict, save_examples, summarise  # noqa: E402


def headline(res: dict, task: str) -> dict:
    if task == "segmentation":
        return {c: {"mean_dice": v["mean_dice"], "ci95": v["dice_ci_patient_bootstrap"]["ci95"],
                    "n_patients": v["dice_ci_patient_bootstrap"]["n_clusters"], "missed": v["n_missed_entirely"],
                    "false_alarm": v["n_false_alarm_images"]} for c, v in res["per_class"].items()}
    return {"balanced_accuracy": res["balanced_accuracy"], "macro_f1": res["macro_f1"],
            "accuracy_ci95": res["accuracy_ci_patient_bootstrap"]["ci95"],
            "n_patients": res["accuracy_ci_patient_bootstrap"]["n_clusters"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--name", required=True, help="label for this evaluation set, e.g. external_hospitalB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="")
    ap.add_argument("--internal", default="", help="metrics.json of the internal run for comparison")
    ap.add_argument("--base", default="")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ck = torch.load(args.checkpoint, map_location=dev)
    cfg = ck["config"]
    task, classes, size = cfg["task_type"], cfg["class_names"], ck["input_size"]
    model = build_model(task, len(classes), False, cfg.get("training", {}).get("arch")).to(dev)
    model.load_state_dict(ck["state_dict"])
    rows = usable_rows(read_manifest(args.manifest), args.split or None)
    if not rows:
        sys.exit("[STOP] no annotated rows in the evaluation manifest")
    base = Path(args.base) if args.base else Path(args.manifest).resolve().parent
    ds = FrameSet(rows, task, size, base, train=False)
    preds = predict(model, DataLoader(ds, batch_size=8), task, dev)
    res, per_case = summarise(preds, rows, task, classes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_case.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_case[0]))
        w.writeheader(), w.writerows(per_case)
    save_examples(preds, rows, ds, classes, out / "examples", task)
    report = {"evaluation_set": args.name, "checkpoint": str(args.checkpoint), "task_type": task,
              "class_names": classes, "arch": cfg.get("training", {}).get("arch"), "n_frames": len(rows),
              "n_patients": len({r["patient_id"] for r in rows}), "result": res, "headline": headline(res, task),
              "retrained_or_tuned": False}
    if args.internal:
        import json
        im = json.loads(Path(args.internal).read_text())
        if im.get("test"):
            report["internal_headline"] = headline(im["test"], task)
    write_json(report, out / "external_metrics.json")
    lines = [f"| Set | {'Class | Mean Dice (95% CI) | Missed | False-alarm' if task == 'segmentation' else 'Balanced acc | Macro F1 | Accuracy 95% CI'} | Patients |",
             "|---|---|---|---|---|" + ("---|" if task == "segmentation" else "")]
    for label, h in (("internal test", report.get("internal_headline")), (args.name, report["headline"])):
        if not h:
            continue
        if task == "segmentation":
            for c, v in h.items():
                lines.append(f"| {label} | {c} | {v['mean_dice']:.3f} ({v['ci95'][0]:.3f}–{v['ci95'][1]:.3f}) | "
                             f"{v['missed']} | {v['false_alarm']} | {v['n_patients']} |")
        else:
            ci = h["accuracy_ci95"]
            lines.append(f"| {label} | {h['balanced_accuracy']:.3f} | {h['macro_f1']:.3f} | {ci[0]:.3f}–{ci[1]:.3f} | {h['n_patients']} |")
    (out / "table3_external.md").write_text("\n".join(lines) + "\n\nThe checkpoint was frozen; no retraining or tuning on this set.\n",
                                            encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
