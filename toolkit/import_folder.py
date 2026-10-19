"""Turn an already-labelled image folder (e.g. a public dataset) into a manifest.

Works for any procedure as long as you can say, with regular expressions, how to
read the patient/video ID from a file path, and where each image's label lives.

  segmentation   masks as index PNGs (pixel value = class index) or colour PNGs
                 (--colors "liver=255,0,0;gallbladder=0,255,0"), matched to the
                 image by replacing --image-dir with --mask-dir and an optional suffix
  classification a CSV with columns image,label  (path relative to --root)

Usage:
  python import_folder.py --root data/ --images "images/**/*.png" --video-regex "(video\\d+)" \
      --config config.json --mask-dir images:masks --out labeled/
  python import_folder.py --root data/ --images "**/*.jpg" --video-regex "(case\\d+)" \
      --config config.json --labels-csv labels.csv --out labeled/
Patient ID = video ID unless --patient-regex is given. Check the split later:
two videos of the same patient must share a patient ID.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sv_common import IGNORE_INDEX, load_config, write_json, write_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--images", required=True, help="glob relative to root")
    ap.add_argument("--video-regex", required=True)
    ap.add_argument("--patient-regex", default="")
    ap.add_argument("--frame-regex", default=r"(\d+)(?!.*\d)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mask-dir", default="", help="'imgdir:maskdir' path substitution for masks")
    ap.add_argument("--mask-suffix", default="", help="e.g. _mask (inserted before extension)")
    ap.add_argument("--colors", default="", help="colour masks: 'name=r,g,b;...'")
    ap.add_argument("--labels-csv", default="")
    ap.add_argument("--fps", type=float, default=0, help="source fps, to fill time_sec (optional)")
    ap.add_argument("--source-tag", default="public_dataset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    classes = cfg["class_names"]
    root, out = Path(args.root).resolve(), Path(args.out)
    imgs = sorted(root.glob(args.images))
    if not imgs:
        sys.exit("[STOP] no images matched --images")
    labels = {}
    if args.labels_csv:
        with open(args.labels_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["label"] not in classes:
                    sys.exit(f"[STOP] label {r['label']!r} not in config.class_names")
                labels[str((root / r["image"]).resolve())] = classes.index(r["label"])
    colors = {}
    for item in filter(None, args.colors.split(";")):
        name, rgb = item.split("=")
        if name not in classes:
            sys.exit(f"[STOP] colour class {name!r} not in config.class_names")
        colors[name] = tuple(int(v) for v in rgb.split(","))
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], {"no_video_id": 0, "no_label": 0}
    for p in imgs:
        rel = str(p.relative_to(root))
        mv = re.search(args.video_regex, rel)
        if not mv:
            skipped["no_video_id"] += 1
            continue
        vid = mv.group(1)
        pid = re.search(args.patient_regex, rel).group(1) if args.patient_regex else vid
        mf = re.search(args.frame_regex, p.stem)
        fidx = int(mf.group(1)) if mf else 0
        row = {"sample_id": f"{vid}_{p.stem}", "patient_id": pid, "video_id": vid, "frame_idx": fidx,
               "time_sec": round(fidx / args.fps, 3) if args.fps else "", "image_path": str(p),
               "mask_path": "", "label": "", "split": "", "annotation_status": "done",
               "label_source": args.source_tag}
        if cfg["task_type"] == "segmentation":
            a, b = (args.mask_dir.split(":") + [""])[:2] if args.mask_dir else ("", "")
            mp = Path(str(p).replace(f"/{a}/", f"/{b}/") if a else str(p))
            mp = mp.with_name(mp.stem + args.mask_suffix + ".png")
            if not mp.exists():
                skipped["no_label"] += 1
                continue
            m = Image.open(mp)
            if colors:
                rgb = np.asarray(m.convert("RGB"))
                idx = np.zeros(rgb.shape[:2], np.uint8)
                for name, col in colors.items():
                    idx[np.all(rgb == col, axis=-1)] = classes.index(name)
            else:
                idx = np.asarray(m)
                bad = set(np.unique(idx)) - set(range(len(classes))) - {IGNORE_INDEX}
                if bad:
                    sys.exit(f"[STOP] {mp.name} has pixel values {sorted(bad)} outside 0..{len(classes)-1}/255 "
                             "— use --colors or remap first")
            dst = out / "masks" / f"{row['sample_id']}.png"
            Image.fromarray(idx.astype(np.uint8)).save(dst)
            row["mask_path"] = str(dst.resolve())
        else:
            lab = labels.get(str(p))
            if lab is None:
                skipped["no_label"] += 1
                continue
            row["label"] = str(lab)
        rows.append(row)
    write_manifest(rows, out / "labeled_manifest.csv")
    rep = {"n_rows": len(rows), "skipped": skipped, "n_videos": len({r["video_id"] for r in rows}),
           "n_patients": len({r["patient_id"] for r in rows})}
    write_json(rep, out / "import_report.json")
    print(rep)


if __name__ == "__main__":
    main()
