"""Assign train/val/test at the PATIENT level and prove there is no leakage.

Frames from one surgery look alike. If the same patient appears in train and
test, the score measures memory of that patient, not generalisation. So whole
patients are assigned to one split. Optional --stratify-col balances a
patient-level attribute (e.g. centre, difficulty grade) across splits.

Usage:
  python make_splits.py --manifest labeled/labeled_manifest.csv --ratios 0.6,0.2,0.2 --seed 42 --out split/
  python make_splits.py ... --test-patients P031,P032   # fixed external/temporal test set
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sv_common import read_manifest, usable_rows, write_json, write_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ratios", default="0.6,0.2,0.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify-col", default="")
    ap.add_argument("--test-patients", default="", help="comma list forced into test")
    ap.add_argument("--keep-existing", action="store_true", help="respect a split column already filled")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    done = usable_rows(rows)
    if not done:
        sys.exit("[STOP] no annotated rows (annotation_status=done)")
    r_tr, r_va, r_te = (float(x) for x in args.ratios.split(","))
    if args.keep_existing and all(r.get("split") for r in done):
        seen = defaultdict(set)
        for r in done:
            seen[r["patient_id"]].add(r["split"])
        mixed = {p: sorted(s) for p, s in seen.items() if len(s) > 1}
        if mixed:
            sys.exit(f"[STOP] existing split puts one patient in several sets (leakage): {mixed}")
        assign = {p: s.pop() for p, s in seen.items()}
    else:
        forced = {p for p in args.test_patients.split(",") if p}
        groups = defaultdict(list)
        strat = {}
        for r in done:
            strat[r["patient_id"]] = r.get(args.stratify_col, "") if args.stratify_col else ""
        for p, s in strat.items():
            if p not in forced:
                groups[s].append(p)
        rng = random.Random(args.seed)
        assign = {p: "test" for p in forced}
        for s, pats in sorted(groups.items()):
            pats = sorted(pats)
            rng.shuffle(pats)
            n = len(pats)
            n_te = 0 if forced else max(1, round(n * r_te)) if n >= 3 else 0
            n_va = max(1, round(n * r_va)) if n - n_te >= 2 else 0
            for i, p in enumerate(pats):
                assign[p] = "test" if i < n_te else "val" if i < n_te + n_va else "train"

    for r in rows:
        r["split"] = assign.get(r["patient_id"], "") if r.get("annotation_status") == "done" else ""

    pats_by_split = defaultdict(set)
    for r in usable_rows(rows):
        pats_by_split[r["split"]].add(r["patient_id"])
    leaks = {a + "&" + b: sorted(pats_by_split[a] & pats_by_split[b])
             for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(leaks.values()):
        sys.exit(f"[STOP] patient leakage: {leaks}")

    summary = {"seed": args.seed, "ratios": [r_tr, r_va, r_te], "unit": "patient_id",
               "patients": {s: sorted(p) for s, p in pats_by_split.items()},
               "n_patients": {s: len(p) for s, p in pats_by_split.items()},
               "n_frames": dict(Counter(r["split"] for r in usable_rows(rows))),
               "leakage": "none", "warnings": []}
    if summary["n_patients"].get("test", 0) < 5:
        summary["warnings"].append("test set has <5 patients: confidence intervals will be unreliable; "
                                   "report as feasibility only")
    if any(r.get("label") for r in usable_rows(rows)):
        summary["label_counts"] = {s: dict(Counter(r["label"] for r in usable_rows(rows, s)))
                                   for s in ("train", "val", "test")}
    out = Path(args.out)
    write_manifest(rows, out / "split_manifest.csv")
    write_json(summary, out / "split_summary.json")
    print({k: summary[k] for k in ("n_patients", "n_frames", "leakage", "warnings")})


if __name__ == "__main__":
    main()
