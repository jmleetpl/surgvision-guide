"""Apply a trained checkpoint to a video (or a folder of frames).

  segmentation   -> overlay.mp4 with the predicted mask(s) blended in
  classification -> predictions.csv + timeline.png (predicted label over time)

Use a video that was NOT in train/val. If it belongs to a test patient it is
the same evidence as the test metrics, not an additional evaluation.

Usage:
  python infer_video.py --checkpoint runs/run01/checkpoint_best.pt --video V031.mp4 --fps 5 --out runs/run01/video
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model  # noqa: E402
from train import MEAN, STD, pick_device  # noqa: E402

PAL = np.array([[0, 0, 0], [0, 220, 0], [255, 0, 255], [0, 160, 255], [255, 200, 0], [255, 60, 60]], np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video", help="video file")
    ap.add_argument("--frames-dir", help="or a folder of PNG/JPG frames (sorted by name)")
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ck = torch.load(args.checkpoint, map_location=dev)
    cfg, size = ck["config"], tuple(ck["input_size"])
    task, classes = cfg["task_type"], cfg["class_names"]
    model = build_model(task, len(classes), False, cfg.get("training", {}).get("arch")).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    if args.video:
        subprocess.run(["ffmpeg", "-v", "error", "-i", args.video, "-map_metadata", "-1", "-vf", f"fps={args.fps}",
                        str(tmp / "f_%06d.png")], check=True)
        src = sorted(tmp.glob("f_*.png"))
    else:
        src = sorted(p for p in Path(args.frames_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not src:
        sys.exit("[STOP] no frames found")

    rows = []
    (tmp / "ov").mkdir()
    with torch.no_grad():
        for i, p in enumerate(src):
            img = Image.open(p).convert("RGB").resize(size, Image.BILINEAR)
            a = np.asarray(img, np.float32)
            x = torch.from_numpy(((a / 255 - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(dev)
            logits = model(x)
            if task == "segmentation":
                m = logits.argmax(1)[0].cpu().numpy()
                alpha = (m > 0)[..., None] * 0.45
                ov = a * (1 - alpha) + PAL[np.clip(m, 0, len(PAL) - 1)] * alpha
                Image.fromarray(ov.astype(np.uint8)).save(tmp / "ov" / f"o_{i:06d}.png")
                rows.append({"frame": i, "time_sec": round(i / args.fps, 3),
                             **{f"px_{c}": int((m == k).sum()) for k, c in enumerate(classes) if k}})
            else:
                pr = logits.softmax(1)[0].cpu().numpy()
                rows.append({"frame": i, "time_sec": round(i / args.fps, 3), "pred": classes[int(pr.argmax())],
                             **{f"p_{c}": round(float(v), 4) for c, v in zip(classes, pr)}})

    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)
    info = {"source": args.video or args.frames_dir, "checkpoint": args.checkpoint, "n_frames": len(rows),
            "fps": args.fps, "task_type": task}
    if task == "segmentation":
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(args.fps), "-i", str(tmp / "ov" / "o_%06d.png"),
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out / "overlay.mp4")], check=True)
        info["overlay"] = "overlay.mp4"
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = [r["time_sec"] for r in rows]
        y = [classes.index(r["pred"]) for r in rows]
        fig, ax = plt.subplots(figsize=(10, 2.4))
        ax.scatter(t, y, s=6)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(out / "timeline.png", dpi=150)
        info["timeline"] = "timeline.png"
    (out / "inference_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
