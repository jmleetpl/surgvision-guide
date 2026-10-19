"""Turn a run folder into manuscript-ready tables and figures.

Reads ONLY files produced by this pipeline (split_summary.json, metrics.json,
history.csv, per_case_test.csv). Writes:
  results/table1_dataset.md    patients / videos / frames per split (+ label counts)
  results/table2_performance.md test metrics with patient-level bootstrap CI
  results/fig_learning_curve.png
  results/fig_per_case.png     per-image Dice distribution (seg) or confusion matrix (cls)
  results/numbers.json          every number the manuscript is allowed to quote

Usage:
  python make_report.py --split split/split_summary.json --run runs/run01 --out runs/run01/results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def ci(c):
    if not c or c.get("estimate") is None:
        return "—"
    s = f"{c['estimate']:.3f} ({c['ci95'][0]:.3f}–{c['ci95'][1]:.3f})"
    return s + (" †" if not c.get("reliable") else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = json.loads(Path(args.split).read_text())
    m = json.loads((run / "metrics.json").read_text())
    task, classes = m["task_type"], m["class_names"]

    t1 = ["| Split | Patients | Frames |" + (" Label counts |" if sp.get("label_counts") else ""),
          "|---|---:|---:|" + ("---|" if sp.get("label_counts") else "")]
    for s in ("train", "val", "test"):
        lc = ""
        if sp.get("label_counts"):
            lc = " " + ", ".join(f"{classes[int(k)]}={v}" for k, v in sorted(sp["label_counts"].get(s, {}).items())) + " |"
        t1.append(f"| {s} | {sp['n_patients'].get(s, 0)} | {sp['n_frames'].get(s, 0)} |{lc}")
    t1.append(f"\nSplit unit: {sp['unit']}; seed {sp['seed']}; leakage check: {sp['leakage']}.")
    t1 += [f"\n⚠ {w}" for w in sp.get("warnings", [])]
    (out / "table1_dataset.md").write_text("\n".join(t1), encoding="utf-8")

    te = m.get("test")
    t2 = []
    if te is None:
        t2.append("No test split. Report validation results only and say so.")
    elif task == "segmentation":
        t2 += ["| Class | Mean Dice (95% CI) | Median Dice | Mean IoU | Images scored | Missed entirely | False-alarm images |",
               "|---|---|---:|---:|---:|---:|---:|"]
        for c, v in te["per_class"].items():
            t2.append(f"| {c} | {ci(v['dice_ci_patient_bootstrap'])} | {f(v['median_dice'])} | {f(v['mean_iou'])} | "
                      f"{v['n_images_scored']} | {v['n_missed_entirely']} | {v['n_false_alarm_images']} |")
    else:
        t2 += ["| Metric | Value |", "|---|---|",
               f"| Accuracy (95% CI) | {ci(te['accuracy_ci_patient_bootstrap'])} |",
               f"| Balanced accuracy | {f(te['balanced_accuracy'])} |", f"| Macro F1 | {f(te['macro_f1'])} |"]
        t2 += ["", "| Class | Recall | Precision | F1 | AUROC (one-vs-rest, pooled) |", "|---|---:|---:|---:|---:|"]
        au = te.get("per_class_auroc_ovr") or [None] * len(classes)
        for i, c in enumerate(classes):
            t2.append(f"| {c} | {f(te['per_class_recall'][i])} | {f(te['per_class_precision'][i])} | {f(te['per_class_f1'][i])} | {f(au[i])} |")
    t2.append(f"\nModel selected on validation ({m['selection_metric']}) at epoch {m['selected_epoch']} of {m['epochs']}; "
              f"test evaluated once afterwards. CI: patient-level cluster bootstrap. † fewer than 5 test patients — CI not reliable.")
    if m.get("smoke_test"):
        t2.append("\n⚠ SMOKE TEST RUN (limited data or random encoder) — do not report these numbers.")
    (out / "table2_performance.md").write_text("\n".join(t2), encoding="utf-8")

    with open(run / "history.csv") as fh:
        h = list(csv.DictReader(fh))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ep = [int(r["epoch"]) for r in h]
    ax[0].plot(ep, [float(r["train_loss"]) for r in h], marker="o"), ax[0].set_title("train loss")
    ax[1].plot(ep, [float(r["val_score"]) for r in h], marker="o"), ax[1].set_title(f"validation: {m['selection_metric']}")
    ax[1].axvline(m["selected_epoch"], ls="--", c="grey")
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout(), fig.savefig(out / "fig_learning_curve.png", dpi=150), plt.close(fig)

    if te is not None and (run / "per_case_test.csv").exists():
        with open(run / "per_case_test.csv") as fh:
            pc = list(csv.DictReader(fh))
        fig, ax = plt.subplots(figsize=(5, 3.2))
        if task == "segmentation":
            for c in te["per_class"]:
                vals = [float(r[f"dice_{c}"]) for r in pc if r[f"dice_{c}"] not in ("", "None")]
                ax.hist(vals, bins=20, range=(0, 1), alpha=0.6, label=c)
            ax.set_xlabel("per-image Dice (test)"), ax.set_ylabel("images"), ax.legend()
        else:
            cm = np.array(te["confusion_matrix"])
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right"), ax.set_yticks(range(len(classes)), classes)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
            ax.set_xlabel("predicted"), ax.set_ylabel("true")
        fig.tight_layout(), fig.savefig(out / "fig_per_case.png", dpi=150), plt.close(fig)

    numbers = {"split": sp, "metrics": m}
    (out / "numbers.json").write_text(json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(t1) + "\n\n" + "\n".join(t2))


if __name__ == "__main__":
    main()
