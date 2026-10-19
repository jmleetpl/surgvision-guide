"""How many cases are enough? Two complementary answers.

1) precision  (analytic — how many TEST patients)
   Choose how wide the 95% CI of the primary metric may be. Frames inside one
   patient are correlated, so the effective sample size shrinks by the design
   effect DE = 1 + (m - 1) * ICC (m = frames per patient). With a pilot
   per_case CSV the SD and ICC are estimated from your own data.

   python sample_size.py precision --metric dice --expected 0.80 --sd 0.15 --frames-per-patient 20 \
       --icc 0.3 --half-width 0.05
   python sample_size.py precision --metric dice --pilot runs/pilot/per_case_test.csv --column dice_liver \
       --frames-per-patient 20 --half-width 0.05
   python sample_size.py precision --metric proportion --expected 0.85 --frames-per-patient 15 --icc 0.2 \
       --half-width 0.07      # sensitivity / accuracy / balanced accuracy component

2) curve  (empirical — how many TRAINING patients)
   Retrains on growing subsets of training patients (validation only, test
   untouched), fits an inverse power law  score(n) = a - b * n^(-c)
   (Figueroa et al., BMC Med Inform Decis Mak 2012) and extrapolates the number
   of patients needed to reach --target, or 95% of the fitted plateau a.
   Treat the extrapolation as a planning estimate; with < 4 points it is fragile.

   python sample_size.py curve --config config.json --manifest split/split_manifest.csv --out lc/ \
       --steps 1,2,3 --reps 2 --epochs 4 [--target 0.85]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sv_common import read_manifest, usable_rows, write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
Z = 1.959964


def icc_oneway(groups: list[list[float]]) -> float:
    """ICC(1) from one-way ANOVA with unequal group sizes."""
    groups = [g for g in groups if len(g) > 0]
    k, n = len(groups), sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return float("nan")
    grand = np.mean([x for g in groups for x in g])
    msb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((np.array(g) - np.mean(g)) ** 2).sum() for g in groups) / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    return float(max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))) if msb + (n0 - 1) * msw > 0 else 0.0


def precision(args) -> dict:
    sd, icc, src = args.sd, args.icc, "assumed"
    if args.pilot:
        import csv
        by = defaultdict(list)
        with open(args.pilot, newline="") as f:
            for r in csv.DictReader(f):
                v = r.get(args.column, "")
                if v not in ("", "None"):
                    by[r["patient_id"]].append(float(v))
        vals = [x for g in by.values() for x in g]
        if len(vals) < 5:
            sys.exit("[STOP] pilot has < 5 scored frames")
        sd, icc, src = float(np.std(vals, ddof=1)), icc_oneway(list(by.values())), f"pilot ({len(by)} patients, {len(vals)} frames)"
        if math.isnan(icc):
            icc, src = args.icc if args.icc is not None else 0.3, src + "; ICC not estimable (1 patient) → assumed"
        args.expected = args.expected or float(np.mean(vals))
    if args.metric == "proportion":
        p = args.expected
        sd = math.sqrt(p * (1 - p))
    if sd is None or icc is None:
        sys.exit("[STOP] give --sd and --icc, or --pilot")
    m = args.frames_per_patient
    de = 1 + (m - 1) * icc
    n_frames_srs = (Z * sd / args.half_width) ** 2
    n_frames = n_frames_srs * de
    n_pat = math.ceil(n_frames / m)
    res = {"metric": args.metric, "expected": args.expected, "sd": round(sd, 4), "icc": round(icc, 3),
           "source": src, "frames_per_patient": m, "design_effect": round(de, 2), "half_width": args.half_width,
           "frames_needed": math.ceil(n_frames), "test_patients_needed": max(n_pat, 5),
           "note": "minimum 5 patients enforced; more frames per patient helps little when ICC is high"}
    # what if: frames per patient sensitivity
    res["what_if_frames_per_patient"] = {mm: max(5, math.ceil(n_frames_srs * (1 + (mm - 1) * icc) / mm))
                                         for mm in (5, 10, 20, 40, 80)}
    if args.metric == "proportion" and args.prevalence:
        res["note_prevalence"] = (f"for sensitivity, these counts are POSITIVE frames; with prevalence "
                                  f"{args.prevalence} you need ~{math.ceil(res['frames_needed'] / args.prevalence)} frames in total")
    return res


def power_law(n, a, b, c):
    return a - b * np.power(n, -c)


def curve(args) -> dict:
    rows = read_manifest(args.manifest)
    train_p = sorted({r["patient_id"] for r in usable_rows(rows, "train")})
    steps = [int(s) for s in args.steps.split(",")] if args.steps else sorted({max(1, round(len(train_p) * f)) for f in (0.25, 0.5, 0.75, 1.0)})
    steps = [s for s in steps if s <= len(train_p)]
    if len(steps) < 2:
        sys.exit(f"[STOP] need >= 2 subset sizes; only {len(train_p)} training patients")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    points = []
    for n in steps:
        reps = 1 if n == len(train_p) else args.reps
        for rep in range(reps):
            sub = rng.sample(train_p, n)
            run = out / f"n{n}_rep{rep}"
            cmd = [sys.executable, str(HERE / "train.py"), "--config", args.config, "--manifest", args.manifest,
                   "--out", str(run), "--epochs", str(args.epochs), "--no-test", "--train-patients", ",".join(sub)]
            cmd += (["--base", args.base] if args.base else []) + (["--no-pretrained"] if args.no_pretrained else [])
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"n={n} rep={rep} failed: {r.stderr.strip().splitlines()[-1:]}")
                continue
            m = json.loads((run / "metrics.json").read_text())
            points.append({"n_train_patients": n, "rep": rep, "n_train_frames": m["n_frames"]["train"],
                           "val_score": m["val_selected"]})
            print(points[-1], flush=True)
    xs = np.array([p["n_train_patients"] for p in points], float)
    ys = np.array([p["val_score"] for p in points], float)
    res = {"points": points, "metric": "validation selection score", "fit": None}
    if len(set(xs)) >= 3:
        try:
            from scipy.optimize import curve_fit
            (a, b, c), _ = curve_fit(power_law, xs, ys, p0=[min(1.0, ys.max() + 0.05), 0.5, 0.5],
                                     bounds=([0, 0, 0.01], [1.0, 10, 5]), maxfev=20000)
            target = args.target or 0.95 * a
            need = None if target >= a else math.ceil(((a - target) / b) ** (-1 / c))
            res["fit"] = {"plateau_a": round(a, 4), "b": round(b, 4), "c": round(c, 4), "target": round(target, 4),
                          "train_patients_needed": need,
                          "warning": "extrapolation beyond observed sizes is uncertain" if need and need > xs.max() else ""}
        except Exception as e:  # noqa: BLE001
            res["fit_error"] = str(e)
    else:
        res["fit_error"] = "need >= 3 distinct subset sizes to fit the curve"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.scatter(xs, ys, label="observed (validation)")
        if res["fit"]:
            g = np.linspace(1, max(xs.max() * 3, (res["fit"]["train_patients_needed"] or 0) * 1.1, 4), 100)
            ax.plot(g, power_law(g, res["fit"]["plateau_a"], res["fit"]["b"], res["fit"]["c"]), label="inverse power-law fit")
            ax.axhline(res["fit"]["target"], ls="--", c="grey", label="target")
        ax.set_xlabel("training patients"), ax.set_ylabel("validation score"), ax.legend(fontsize=7)
        fig.tight_layout(), fig.savefig(out / "learning_curve.png", dpi=150)
    except Exception:  # noqa: BLE001
        pass
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precision")
    p.add_argument("--metric", choices=["dice", "proportion"], required=True)
    p.add_argument("--expected", type=float)
    p.add_argument("--sd", type=float)
    p.add_argument("--icc", type=float)
    p.add_argument("--pilot", default="")
    p.add_argument("--column", default="")
    p.add_argument("--frames-per-patient", type=int, required=True)
    p.add_argument("--half-width", type=float, required=True)
    p.add_argument("--prevalence", type=float)
    p.add_argument("--out", default="")
    c = sub.add_parser("curve")
    for a in ("--config", "--manifest", "--out"):
        c.add_argument(a, required=True)
    c.add_argument("--steps", default="")
    c.add_argument("--reps", type=int, default=2)
    c.add_argument("--epochs", type=int, default=4)
    c.add_argument("--target", type=float)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--base", default="")
    c.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    res = precision(args) if args.cmd == "precision" else curve(args)
    if getattr(args, "out", ""):
        write_json(res, Path(args.out) / f"sample_size_{args.cmd}.json")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
