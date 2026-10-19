"""Extract frames from surgical videos with de-identification steps.

What it does (and what it does NOT do):
  * renames every video to a study ID from a linkage CSV (original names never
    leave this step; keep the linkage file inside the hospital),
  * strips container metadata (ffmpeg -map_metadata -1),
  * optionally blacks out fixed rectangles where the recorder burns in text,
  * samples frames at --fps and writes frames_manifest.csv,
  * flags candidate out-of-body frames with a simple colour heuristic
    (laparoscopic/robotic/endoscopic only; off for microscopic/open video) and
    writes contact sheets for a HUMAN to review.
  The heuristic is a screening aid only. A person must review the contact
  sheets and mark out-of-body frames before anything leaves the hospital.

Usage:
  python extract_frames.py --videos raw_videos/ --linkage linkage.csv --out frames/ --fps 1
  linkage.csv columns: original_file,patient_id,video_id   (e.g. case_017.mp4,P017,V017)
  --blackout "x,y,w,h;x,y,w,h"   pixel rectangles in the source resolution
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sv_common import write_json, write_manifest  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv"}


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    st = info["streams"][0]
    num, den = (st.get("r_frame_rate") or "0/1").split("/")
    return {"width": int(st["width"]), "height": int(st["height"]),
            "fps": float(num) / float(den) if float(den) else None,
            "duration_sec": float(info.get("format", {}).get("duration") or 0)}


def oob_score(img: Image.Image) -> float:
    """Higher = more likely out-of-body. Inside the abdomen frames are dominated by
    red/pink tissue; OR-room scenes are greyer/bluer/brighter. Screening only."""
    a = np.asarray(img.convert("RGB").resize((64, 36)), dtype=np.float32) / 255
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    redness = float(np.mean(r - (g + b) / 2))
    brightness = float(a.mean())
    return float(np.clip(0.5 - 2.0 * redness + 0.6 * max(0.0, brightness - 0.55), 0, 1))


def contact_sheet(rows: list[dict], root: Path, out: Path, cols: int = 8, thumb=(192, 108)):
    sheets = []
    for start in range(0, len(rows), cols * 6):
        chunk = rows[start:start + cols * 6]
        sheet = Image.new("RGB", (cols * thumb[0], ((len(chunk) - 1) // cols + 1) * (thumb[1] + 14)), "white")
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk):
            im = Image.open(root / r["image_path"]).convert("RGB").resize(thumb)
            x, y = (i % cols) * thumb[0], (i // cols) * (thumb[1] + 14)
            sheet.paste(im, (x, y))
            flag = "OOB?" if r["oob_flag"] == "1" else ""
            d.text((x + 2, y + thumb[1]), f"{r['sample_id'][-14:]} {flag}", fill="red" if flag else "black")
        p = out / f"contact_{len(sheets):03d}.jpg"
        sheet.save(p, quality=85)
        sheets.append(p.name)
    return sheets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--linkage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-width", type=int, default=854, help="downscale frames to this width")
    ap.add_argument("--blackout", default="", help='"x,y,w,h;..." source-pixel rectangles to black out')
    ap.add_argument("--oob-threshold", type=float, default=0.55)
    ap.add_argument("--modality", default="laparoscopic",
                    choices=["laparoscopic", "robotic", "endoscopic", "microscopic", "open"],
                    help="colour out-of-body screening is only meaningful inside a body cavity; "
                         "for microscopic/open video it is switched off and every frame relies on human review")
    args = ap.parse_args()

    vids, out = Path(args.videos), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(exist_ok=True)
    with open(args.linkage, newline="", encoding="utf-8") as f:
        link = {r["original_file"]: r for r in csv.DictReader(f)}

    use_heuristic = args.modality in ("laparoscopic", "robotic", "endoscopic")
    rects = [tuple(int(v) for v in s.split(",")) for s in args.blackout.split(";") if s.strip()]
    rows, report = [], {"videos": [], "fps": args.fps, "blackout": rects,
                        "metadata_stripped": True, "oob_threshold": args.oob_threshold,
                        "modality": args.modality}
    files = sorted(p for p in vids.iterdir() if p.suffix.lower() in VIDEO_EXT)
    missing = [p.name for p in files if p.name not in link]
    if missing:
        sys.exit(f"[STOP] linkage.csv has no study ID for: {missing}. Add them; do not use raw names.")

    for p in files:
        meta, lk = probe(p), link[p.name]
        vid, pid = lk["video_id"], lk["patient_id"]
        vdir = out / "images" / vid
        vdir.mkdir(parents=True, exist_ok=True)
        filt = [f"drawbox=x={x}:y={y}:w={w}:h={h}:color=black:t=fill" for x, y, w, h in rects]
        filt += [f"fps={args.fps}", f"scale='min({args.max_width},iw)':-2"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(p), "-map_metadata", "-1",
                        "-vf", ",".join(filt), str(vdir / f"{vid}_%06d.png")], check=True)
        frames = sorted(vdir.glob(f"{vid}_*.png"))
        for i, fp in enumerate(frames):
            score = oob_score(Image.open(fp)) if use_heuristic else 0.0
            rows.append({"sample_id": fp.stem, "patient_id": pid, "video_id": vid,
                         "frame_idx": i, "time_sec": round(i / args.fps, 3),
                         "image_path": str(fp.relative_to(out)), "mask_path": "", "label": "",
                         "split": "", "annotation_status": "not_started",
                         "oob_score": round(score, 3), "oob_flag": "1" if score >= args.oob_threshold else "0",
                         "oob_reviewed": "0"})
        report["videos"].append({"video_id": vid, "patient_id": pid, "source_fps": meta["fps"],
                                 "duration_sec": meta["duration_sec"], "source_size": [meta["width"], meta["height"]],
                                 "n_frames": len(frames),
                                 "n_oob_flagged": sum(r["oob_flag"] == "1" for r in rows if r["video_id"] == vid)})

    write_manifest(rows, out / "frames_manifest.csv")
    report["contact_sheets"] = contact_sheet(rows, out, out / "review")
    report["n_frames"] = len(rows)
    report["oob_heuristic"] = "on" if use_heuristic else "off (modality %s) — review every frame for faces, room, screens, paper" % args.modality
    report["next_step"] = ("Review review/contact_*.jpg. For every out-of-body or identifiable frame set "
                           "oob_flag=1 and oob_reviewed=1 in frames_manifest.csv (or delete it). "
                           "Only then run select_frames.py.")
    write_json(report, out / "extract_report.json")
    print(json.dumps({k: report[k] for k in ("n_frames", "contact_sheets", "next_step")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
