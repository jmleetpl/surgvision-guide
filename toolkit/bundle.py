"""Pack toolkit + config + split + referenced images/masks into study_bundle.zip.

Paths inside split_manifest.csv are rewritten to be relative to the bundle,
so the same notebook runs in Colab or on a hospital PC. Only annotated rows
(annotation_status=done) and their files are included. The linkage file and
raw videos are never bundled.

Usage:
  python bundle.py --config config.json --split split/ --out pack/study_bundle.zip [--governance governance_decision.json] [--demo-video V.mp4]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sv_common import read_manifest, write_manifest  # noqa: E402

TOOLKIT = ["sv_common.py", "train.py", "infer_video.py", "make_report.py"]
FORBIDDEN = ("linkage", "original_file")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, help="folder with split_manifest.csv and split_summary.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--governance")
    ap.add_argument("--demo-video")
    args = ap.parse_args()
    here, sp = Path(__file__).resolve().parent, Path(args.split)
    rows = [r for r in read_manifest(sp / "split_manifest.csv") if r.get("annotation_status") == "done"]
    if any(k in r for r in rows[:1] for k in FORBIDDEN):
        sys.exit("[STOP] manifest contains linkage columns — remove them before bundling")
    tmp = Path(tempfile.mkdtemp())
    for f in TOOLKIT:
        (tmp / "toolkit").mkdir(exist_ok=True)
        shutil.copy(here / f, tmp / "toolkit" / f)
    shutil.copy(args.config, tmp / "config.json")
    for r in rows:
        for col, sub in (("image_path", "images"), ("mask_path", "masks")):
            if r.get(col):
                src = Path(r[col])
                if not src.is_absolute():
                    src = (sp / src).resolve()
                dst = Path(sub) / r["video_id"] / src.name
                (tmp / dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, tmp / dst)
                r[col] = str(dst)
    (tmp / "split").mkdir()
    write_manifest(rows, tmp / "split" / "split_manifest.csv")
    shutil.copy(sp / "split_summary.json", tmp / "split" / "split_summary.json")
    if args.governance:
        shutil.copy(args.governance, tmp / "governance_decision.json")
    if args.demo_video:
        shutil.copy(args.demo_video, tmp / "demo_video.mp4")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(tmp))
    shutil.rmtree(tmp)
    print(f"bundle -> {args.out}  ({len(rows)} annotated frames)")


if __name__ == "__main__":
    main()
