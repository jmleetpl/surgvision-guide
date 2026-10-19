"""Compare candidate architectures on VALIDATION only, then pick one.

Each candidate is trained with the same budget and judged on the validation
split (or on patient-level K-fold cross-validation over train+val with --folds K).
The test split is never opened here. The winner is written into a new config;
train.py then trains it once and evaluates test once.

Rule for choosing: highest mean validation score; if a smaller/faster model is
within --tolerance of the best, prefer the smaller one (simpler, cheaper, less
overfitting risk on small surgical datasets).

Usage:
  python model_search.py --config config.json --manifest split/split_manifest.csv --out search/ \
      --candidates unet_resnet18,lraspp_mobilenet,deeplabv3_mobilenet --epochs 6 [--folds 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import CLS_ARCHS, SEG_ARCHS, build_model, count_params  # noqa: E402
from sv_common import load_config, read_manifest, usable_rows, write_json, write_manifest  # noqa: E402

HERE = Path(__file__).resolve().parent


def run(cfg_path, manifest, out, arch, epochs, extra):
    cmd = [sys.executable, str(HERE / "train.py"), "--config", str(cfg_path), "--manifest", str(manifest),
           "--out", str(out), "--arch", arch, "--epochs", str(epochs), "--no-test"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr.strip().splitlines()[-1:] or ["failed"]
    m = json.loads((Path(out) / "metrics.json").read_text())
    return m, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", default="")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--folds", type=int, default=1, help="1 = fixed val split; K>1 = patient K-fold on train+val")
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base", default="")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    zoo = SEG_ARCHS if cfg["task_type"] == "segmentation" else CLS_ARCHS
    cands = [c for c in args.candidates.split(",") if c] or list(zoo)[:3]
    bad = [c for c in cands if c not in zoo]
    if bad:
        sys.exit(f"[STOP] unknown candidates {bad}; choose from {list(zoo)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = (["--base", args.base] if args.base else []) + (["--no-pretrained"] if args.no_pretrained else []) \
        + (["--limit", str(args.limit)] if args.limit else [])

    rows = read_manifest(args.manifest)
    folds = []
    if args.folds <= 1:
        folds.append(Path(args.manifest))
    else:
        dev_rows = [r for r in usable_rows(rows) if r["split"] in ("train", "val")]
        pats = sorted({r["patient_id"] for r in dev_rows})
        if len(pats) < args.folds:
            sys.exit(f"[STOP] {len(pats)} development patients < {args.folds} folds")
        random.Random(args.seed).shuffle(pats)
        for k in range(args.folds):
            val_p = set(pats[k::args.folds])
            fr = []
            for r in rows:
                r2 = dict(r)
                if r2.get("annotation_status") == "done" and r2.get("split") in ("train", "val"):
                    r2["split"] = "val" if r2["patient_id"] in val_p else "train"
                elif r2.get("split") == "test":
                    r2["split"] = ""            # test is hidden from the search
                fr.append(r2)
            fp = out / f"fold{k}_manifest.csv"
            write_manifest(fr, fp)
            folds.append(fp)

    board = []
    for arch in cands:
        scores, err = [], None
        for k, fm in enumerate(folds):
            m, err = run(args.config, fm, out / arch / f"fold{k}", arch, args.epochs, extra)
            if m is None:
                break
            scores.append(m["val_selected"])
        n_params = count_params(build_model(cfg["task_type"], len(cfg["class_names"]), False, arch))
        board.append({"arch": arch, "description": zoo[arch], "params_M": round(n_params / 1e6, 1),
                      "val_scores": scores, "val_mean": round(sum(scores) / len(scores), 4) if scores and not err else None,
                      "error": err})
        print(board[-1], flush=True)

    ok = [b for b in board if b["val_mean"] is not None]
    if not ok:
        sys.exit("[STOP] every candidate failed")
    best = max(ok, key=lambda b: b["val_mean"])
    near = [b for b in ok if best["val_mean"] - b["val_mean"] <= args.tolerance]
    chosen = min(near, key=lambda b: b["params_M"])
    new_cfg = dict(cfg)
    new_cfg["training"] = dict(cfg.get("training", {}), arch=chosen["arch"])
    (out / "config_selected.json").write_text(json.dumps(new_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "params_M", "val_mean", "val_scores", "error", "description"])
        w.writeheader()
        for b in sorted(board, key=lambda b: -(b["val_mean"] or -1)):
            w.writerow(b)
    summary = {"selection_metric": "validation " + ("mean foreground Dice" if cfg["task_type"] == "segmentation" else "macro F1"),
               "folds": args.folds, "epochs_per_candidate": args.epochs, "tolerance": args.tolerance,
               "best_by_score": best["arch"], "chosen": chosen["arch"],
               "chosen_reason": "highest validation score" if chosen is best else
               f"within {args.tolerance} of best ({best['arch']}) and smaller",
               "test_split_used": False, "leaderboard": board}
    write_json(summary, out / "search_summary.json")
    print(json.dumps({k: summary[k] for k in ("best_by_score", "chosen", "chosen_reason")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
