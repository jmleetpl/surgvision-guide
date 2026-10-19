"""Choose which extracted frames to annotate and package them for CVAT.

Near-identical neighbouring frames add annotation cost without adding
information, so frames are spread across each video with a minimum time gap.
Out-of-body frames (oob_flag=1) are always excluded. The script refuses to run
until a human has reviewed the out-of-body screening (oob_reviewed=1 everywhere)
unless --allow-unreviewed is given for a public-data dry run.

Each video becomes its own CVAT batch ZIP, because frames from different
videos are not a continuous sequence (never use CVAT tracking/interpolation
across them).

Usage:
  python select_frames.py --frames frames/ --per-video 40 --min-gap-sec 5 --out annotate/
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sv_common import read_manifest, write_json, write_manifest  # noqa: E402


def spread(rows: list[dict], k: int, min_gap: float) -> list[dict]:
    rows = sorted(rows, key=lambda r: float(r["time_sec"]))
    if not rows:
        return []
    picked, last = [], -1e9
    step = max(1, len(rows) // max(k, 1))
    for r in rows[::step]:
        t = float(r["time_sec"])
        if t - last >= min_gap:
            picked.append(r)
            last = t
        if len(picked) >= k:
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder with frames_manifest.csv")
    ap.add_argument("--per-video", type=int, default=40)
    ap.add_argument("--min-gap-sec", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unreviewed", action="store_true")
    args = ap.parse_args()

    root, out = Path(args.frames), Path(args.out)
    rows = read_manifest(root / "frames_manifest.csv")
    if not args.allow_unreviewed and any(r.get("oob_reviewed") != "1" for r in rows):
        sys.exit("[STOP] Out-of-body review not finished (oob_reviewed must be 1 for every frame). "
                 "Review the contact sheets first.")
    keep = [r for r in rows if r.get("oob_flag") != "1"]
    by_vid: dict[str, list[dict]] = {}
    for r in keep:
        by_vid.setdefault(r["video_id"], []).append(r)

    out.mkdir(parents=True, exist_ok=True)
    selected, batches = [], []
    for vid, vrows in sorted(by_vid.items()):
        pick = spread(vrows, args.per_video, args.min_gap_sec)
        zpath = out / f"cvat_batch_{vid}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for r in pick:
                z.write(root / r["image_path"], Path(r["image_path"]).name)
        for r in pick:
            r["annotation_status"] = "not_started"
        selected += pick
        batches.append({"video_id": vid, "zip": zpath.name, "n": len(pick)})

    write_manifest(selected, out / "annotation_manifest.csv")
    write_json({"per_video": args.per_video, "min_gap_sec": args.min_gap_sec, "batches": batches,
                "n_selected": len(selected), "excluded_oob": len(rows) - len(keep)}, out / "selection_report.json")
    print(f"selected {len(selected)} frames in {len(batches)} CVAT batches -> {out}")
    for b in batches:
        print(f"  {b['zip']}: {b['n']} frames")


if __name__ == "__main__":
    main()
