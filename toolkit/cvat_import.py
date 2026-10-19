"""Bring CVAT exports (or a phase-interval sheet) back into the manifest.

Modes
  seg       CVAT "Segmentation Mask 1.1" ZIP  -> index masks (0..K-1, 255=ignore)
  tag       CVAT "CVAT for images 1.1" ZIP/XML with frame <tag>s -> label column
  interval  CSV video_id,start_sec,end_sec,label  -> label for every frame inside

A frame counts as annotated ONLY if it carries an annotation (mask pixels of a
class, a tag, or an interval) or is listed in --reviewed-empty (frames a person
checked and confirmed contain none of the classes). Everything else stays
"not_started" and is excluded from training and evaluation. An exported empty
PNG is not a negative label. Frames with oob_flag=1 are never labelled
(status excluded_oob). label_source records which export a label came from;
a later export overrides an earlier one for the frames it contains. Rows already "done" from an earlier export are kept,
so per-video exports can be imported one after another into the same manifest.

Usage:
  python cvat_import.py seg --manifest annotate/annotation_manifest.csv --export my_masks.zip \
      --config config.json --frames frames/ --out labeled/
  python cvat_import.py tag --manifest ... --export tags.zip --config config.json --out labeled/
  python cvat_import.py interval --manifest frames/frames_manifest.csv --intervals phases.csv \
      --config config.json --out labeled/
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sv_common import IGNORE_INDEX, load_config, read_manifest, write_json, write_manifest  # noqa: E402


def parse_labelmap(text: str) -> dict[str, tuple[int, int, int]]:
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, rgb = line.split(":")[0], line.split(":")[1]
        out[name.strip()] = tuple(int(v) for v in rgb.split(","))
    return out


def import_seg(args, cfg, rows):
    classes = cfg["class_names"]                      # index 0 must be background
    z = zipfile.ZipFile(args.export)
    names = z.namelist()
    lm_name = next((n for n in names if n.endswith("labelmap.txt")), None)
    if not lm_name:
        sys.exit("[STOP] labelmap.txt not found — was the export format Segmentation Mask 1.1?")
    lmap = parse_labelmap(z.read(lm_name).decode())
    unknown = [n for n in lmap if n != "background" and n not in classes]
    if unknown:
        sys.exit(f"[STOP] CVAT labels {unknown} are not in config.class_names {classes}")
    masks = {Path(n).stem: n for n in names if "/SegmentationClass/" in "/" + n and n.endswith(".png")}
    reviewed_empty = set(Path(args.reviewed_empty).read_text().split()) if args.reviewed_empty else set()
    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    counts = {"annotated": 0, "reviewed_empty": 0, "not_started": 0}
    for r in rows:
        stem = Path(r["image_path"]).stem
        r["image_path"] = str((Path(args.frames) / r["image_path"]).resolve()) if args.frames else r["image_path"]
        if stem not in masks:
            if r.get("annotation_status") == "done":   # labelled by an earlier export
                counts["kept_from_earlier"] = counts.get("kept_from_earlier", 0) + 1
                continue
            r["annotation_status"] = "not_started"
            counts["not_started"] += 1
            continue
        rgb = np.asarray(Image.open(io.BytesIO(z.read(masks[stem]))).convert("RGB"))
        idx = np.full(rgb.shape[:2], IGNORE_INDEX if args.unlabeled_is_ignore else 0, np.uint8)
        hit = False
        for name, col in lmap.items():
            if name == "background":
                continue
            sel = np.all(rgb == col, axis=-1)
            idx[sel] = classes.index(name)
            hit |= bool(sel.any())
        if not hit and stem not in reviewed_empty:
            if r.get("annotation_status") == "done":   # blank here, labelled earlier: keep earlier
                counts["kept_from_earlier"] = counts.get("kept_from_earlier", 0) + 1
                continue
            r["annotation_status"] = "not_started"      # empty export PNG is not a negative
            counts["not_started"] += 1
            continue
        mp = out / "masks" / f"{stem}.png"
        Image.fromarray(idx).save(mp)
        r["_reimported"] = True                       # a later export overrides earlier labels
        r["mask_path"] = str(mp.resolve())
        r["annotation_status"] = "done"
        counts["annotated" if hit else "reviewed_empty"] += 1
    return counts


def import_tag(args, cfg, rows):
    classes = cfg["class_names"]
    data = Path(args.export).read_bytes()
    if args.export.endswith(".zip"):
        z = zipfile.ZipFile(args.export)
        data = z.read(next(n for n in z.namelist() if n.endswith(".xml")))
    root = ET.fromstring(data)
    tags = {}
    for im in root.iter("image"):
        t = [tg.get("label") for tg in im.findall("tag")]
        if t:
            if len(set(t)) > 1:
                sys.exit(f"[STOP] {im.get('name')} has several tags {t}; one label per frame expected")
            tags[Path(im.get("name")).stem] = t[0]
    counts = {"annotated": 0, "not_started": 0}
    for r in rows:
        lab = tags.get(Path(r["image_path"]).stem)
        if lab is None and r.get("annotation_status") == "done":
            counts["kept_from_earlier"] = counts.get("kept_from_earlier", 0) + 1
        elif lab is None:
            r["annotation_status"] = "not_started"
            counts["not_started"] += 1
        elif lab not in classes:
            sys.exit(f"[STOP] tag {lab!r} not in config.class_names")
        else:
            r["label"], r["annotation_status"], r["_reimported"] = str(classes.index(lab)), "done", True
            counts["annotated"] += 1
    return counts


def import_interval(args, cfg, rows):
    classes = cfg["class_names"]
    iv: dict[str, list] = {}
    with open(args.intervals, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"] not in classes:
                sys.exit(f"[STOP] interval label {r['label']!r} not in config.class_names")
            iv.setdefault(r["video_id"], []).append((float(r["start_sec"]), float(r["end_sec"]), r["label"]))
    counts = {"annotated": 0, "not_started": 0, "overlap_conflict": 0}
    for r in rows:
        t = float(r["time_sec"])
        hits = {lab for s, e, lab in iv.get(r["video_id"], []) if s <= t < e}
        if len(hits) == 1:
            r["label"], r["annotation_status"], r["_reimported"] = str(classes.index(hits.pop())), "done", True
            counts["annotated"] += 1
        else:
            r["annotation_status"] = "not_started"
            counts["overlap_conflict" if hits else "not_started"] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["seg", "tag", "interval"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--export", help="CVAT export ZIP (seg/tag)")
    ap.add_argument("--intervals", help="interval CSV (interval mode)")
    ap.add_argument("--frames", help="folder the manifest image_path is relative to")
    ap.add_argument("--reviewed-empty", help="txt of frame stems confirmed to contain no target class")
    ap.add_argument("--source-tag", default="annotator1",
                    help="who produced these labels (e.g. participant, public_reference); stored in label_source")
    ap.add_argument("--unlabeled-is-ignore", action="store_true",
                    help="seg: pixels left unpainted become 255 (ignore) instead of background")
    args = ap.parse_args()
    cfg, rows = load_config(args.config), read_manifest(args.manifest)
    counts = {"seg": import_seg, "tag": import_tag, "interval": import_interval}[args.mode](args, cfg, rows)
    n_oob = 0
    for r in rows:
        if r.get("oob_flag") == "1":           # out-of-body / identifiable frames are never labelled
            if r.get("annotation_status") == "done":
                n_oob += 1
            r["annotation_status"], r["mask_path"], r["label"] = "excluded_oob", "", ""
        elif r.get("annotation_status") == "done" and r.get("_reimported"):
            r["label_source"] = args.source_tag
        r.pop("_reimported", None)
    counts["excluded_oob"] = n_oob
    if args.mode != "seg" and args.frames:
        for r in rows:
            r["image_path"] = str((Path(args.frames) / r["image_path"]).resolve())
    out = Path(args.out)
    write_manifest(rows, out / "labeled_manifest.csv")
    write_json({"mode": args.mode, "counts": counts, "class_names": cfg["class_names"]}, out / "import_report.json")
    print(counts)


if __name__ == "__main__":
    main()
