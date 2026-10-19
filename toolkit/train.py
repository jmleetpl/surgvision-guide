"""Train, select on validation, then evaluate ONCE on test.

  segmentation   ResNet18-encoder U-Net (encoder optionally ImageNet-pretrained)
  classification ResNet18 frame classifier (phase, CVS criterion, event present ...)

The test split is opened only after the best epoch has been chosen on
validation. Do not change settings after looking at test results; if you do,
the test set has become a second validation set and must be reported as such.

Usage:
  python train.py --config config.json --manifest split/split_manifest.csv --out runs/run01
  python train.py ... --epochs 2 --limit 32 --no-pretrained   # quick smoke test
Outputs: checkpoint_best.pt, history.csv, metrics.json, per_case_test.csv, examples/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import DEFAULT_ARCH, build_model, count_params  # noqa: E402
from sv_common import (IGNORE_INDEX, auroc, cls_metrics, cluster_bootstrap, dice_iou, load_config,  # noqa: E402
                       read_manifest, seed_everything, seg_case_stats, usable_rows, write_json)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def pick_device(req: str) -> torch.device:
    if req == "auto":
        req = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(req)


# ------------------------------------------------------------------ data
class FrameSet(Dataset):
    def __init__(self, rows, task, size, base: Path, train: bool):
        self.rows, self.task, self.size, self.base, self.train = rows, task, tuple(size), base, train

    def __len__(self):
        return len(self.rows)

    def _path(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.base / p

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self._path(r["image_path"])).convert("RGB").resize(self.size, Image.BILINEAR)
        x = (np.asarray(img, np.float32) / 255 - MEAN) / STD
        flip = self.train and np.random.rand() < 0.5
        if self.train:
            x = x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)
        if flip:
            x = x[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        if self.task == "segmentation":
            m = np.asarray(Image.open(self._path(r["mask_path"])).resize(self.size, Image.NEAREST), np.int64)
            if flip:
                m = m[:, ::-1]
            return x, torch.from_numpy(np.ascontiguousarray(m)), i
        return x, torch.tensor(int(r["label"])), i


# ------------------------------------------------------------------ models
# architectures live in models.py (build_model, SEG_ARCHS, CLS_ARCHS)


def seg_loss(logits, y, k):
    ce = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)
    valid = (y != IGNORE_INDEX).unsqueeze(1)
    prob = logits.softmax(1) * valid
    oh = F.one_hot(y.clamp(max=k - 1), k).permute(0, 3, 1, 2) * valid
    inter, union = (prob * oh).sum((0, 2, 3)), prob.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return ce + (1 - ((2 * inter + 1) / (union + 1))[1:].mean())


# ------------------------------------------------------------------ eval
@torch.no_grad()
def predict(model, loader, task, dev):
    model.eval()
    out = []
    for x, y, idx in loader:
        logits = model(x.to(dev))
        pred = logits.argmax(1).cpu().numpy()
        prob = logits.softmax(1).cpu().numpy()
        for j in range(len(idx)):
            out.append((int(idx[j]), y[j].numpy(), pred[j], prob[j] if task == "classification" else None))
    return out


def summarise(preds, rows, task, classes):
    k = len(classes)
    per_case = []
    if task == "segmentation":
        fg = range(1, k)
        by_pat = {c: defaultdict(list) for c in fg}
        for i, gt, pr, _ in preds:
            st = seg_case_stats(pr, gt, k)
            rec = {"sample_id": rows[i]["sample_id"], "patient_id": rows[i]["patient_id"],
                   "video_id": rows[i]["video_id"]}
            for c in fg:
                d, j = dice_iou(st[c])
                rec[f"dice_{classes[c]}"], rec[f"iou_{classes[c]}"] = d, j
                rec[f"gt_px_{classes[c]}"], rec[f"pred_px_{classes[c]}"] = st[c]["gt_px"], st[c]["pred_px"]
                if d is not None:
                    by_pat[c][rows[i]["patient_id"]].append(d)
            per_case.append(rec)
        res = {"per_class": {}}
        for c in fg:
            ds = [r[f"dice_{classes[c]}"] for r in per_case if r[f"dice_{classes[c]}"] is not None]
            js = [r[f"iou_{classes[c]}"] for r in per_case if r[f"iou_{classes[c]}"] is not None]
            res["per_class"][classes[c]] = {
                "mean_dice": float(np.mean(ds)) if ds else None, "median_dice": float(np.median(ds)) if ds else None,
                "mean_iou": float(np.mean(js)) if js else None,
                "n_images_scored": len(ds), "n_both_empty": len(per_case) - len(ds),
                "n_missed_entirely": sum(1 for r in per_case if r[f"gt_px_{classes[c]}"] > 0 and r[f"pred_px_{classes[c]}"] == 0),
                "n_false_alarm_images": sum(1 for r in per_case if r[f"gt_px_{classes[c]}"] == 0 and r[f"pred_px_{classes[c]}"] > 0),
                "dice_ci_patient_bootstrap": cluster_bootstrap(by_pat[c])}
        vals = [v["mean_dice"] for v in res["per_class"].values() if v["mean_dice"] is not None]
        res["selection_score"] = float(np.mean(vals)) if vals else 0.0
        res["selection_metric"] = "mean foreground Dice (per image, then averaged)"
        return res, per_case
    yt, yp = [], []
    by_pat = defaultdict(list)
    for i, gt, pr, prob in preds:
        yt.append(int(gt)), yp.append(int(pr))
        by_pat[rows[i]["patient_id"]].append(float(int(gt) == int(pr)))
        per_case.append({"sample_id": rows[i]["sample_id"], "patient_id": rows[i]["patient_id"],
                         "video_id": rows[i]["video_id"], "time_sec": rows[i].get("time_sec", ""),
                         "true": classes[int(gt)], "pred": classes[int(pr)],
                         **{f"p_{c}": round(float(p), 4) for c, p in zip(classes, prob)}})
    res = cls_metrics(yt, yp, k)
    res["class_names"] = classes
    probs = [p for *_, p in preds]
    res["per_class_auroc_ovr"] = [auroc([int(t == c) for t in yt], [float(p[c]) for p in probs]) for c in range(k)]
    res["accuracy_ci_patient_bootstrap"] = cluster_bootstrap(by_pat)
    per_video = defaultdict(lambda: [[], []])
    for i, gt, pr, _ in preds:
        per_video[rows[i]["video_id"]][0].append(int(gt)), per_video[rows[i]["video_id"]][1].append(int(pr))
    res["per_video_macro_f1"] = {v: cls_metrics(a, b, k)["macro_f1"] for v, (a, b) in per_video.items()}
    res["selection_score"] = res["macro_f1"] or 0.0
    res["selection_metric"] = "macro F1 over frames"
    return res, per_case


def save_examples(preds, rows, ds, classes, out: Path, task, n=6):
    out.mkdir(parents=True, exist_ok=True)
    if task != "segmentation":
        return
    scored = []
    for i, gt, pr, _ in preds:
        d = [dice_iou(s)[0] for s in seg_case_stats(pr, gt, len(classes))[1:]]
        d = [v for v in d if v is not None]
        scored.append((np.mean(d) if d else 1.0, i, gt, pr))
    scored.sort(key=lambda t: t[0])
    pal = np.array([[0, 0, 0], [0, 220, 0], [255, 0, 255], [0, 160, 255], [255, 200, 0], [255, 60, 60]], np.uint8)
    for tag, group in (("worst", scored[:n // 2]), ("best", scored[-(n // 2):])):
        for rank, (d, i, gt, pr) in enumerate(group):
            img = np.asarray(Image.open(ds._path(rows[i]["image_path"])).convert("RGB").resize(ds.size), np.float32)
            panels = [img]
            for m in (gt, pr):
                col = pal[np.clip(m, 0, len(pal) - 1)].astype(np.float32)
                a = ((m > 0) & (m != IGNORE_INDEX))[..., None] * 0.45
                panels.append(img * (1 - a) + col * a)
            Image.fromarray(np.concatenate(panels, 1).astype(np.uint8)).save(
                out / f"{tag}{rank}_{rows[i]['sample_id']}_dice{d:.2f}.png")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="", help="folder relative image/mask paths start from")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, default=0, help="use only N rows per split (smoke test)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arch", default="", help="override config.training.arch (see models.py)")
    ap.add_argument("--train-patients", default="", help="comma list: restrict TRAIN split to these patients "
                    "(learning-curve experiments); val/test unchanged")
    ap.add_argument("--no-test", action="store_true", help="never touch the test split (model search, learning curve)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    task, classes = cfg["task_type"], cfg["class_names"]
    tr = cfg.get("training", {})
    epochs = args.epochs or tr.get("epochs", 10)
    size = tr.get("input_size", [320, 192])
    seed = tr.get("seed", 42)
    seed_everything(seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.base) if args.base else Path(args.manifest).resolve().parent

    rows = read_manifest(args.manifest)
    split = {s: usable_rows(rows, s)[: args.limit or None] for s in ("train", "val", "test")}
    if args.train_patients:
        keep = set(args.train_patients.split(","))
        split["train"] = [r for r in split["train"] if r["patient_id"] in keep]
    if args.no_test:
        split["test"] = []
    for s in ("train", "val"):
        if not split[s]:
            sys.exit(f"[STOP] split '{s}' is empty — run make_splits.py first")
    dev = pick_device(args.device)
    bs, nw = tr.get("batch_size", 8), tr.get("num_workers", 2)
    dl = {s: DataLoader(FrameSet(split[s], task, size, base, s == "train"), batch_size=bs,
                        shuffle=(s == "train"), num_workers=nw,
                        drop_last=(s == "train" and len(split[s]) > bs)) for s in split if split[s]}

    arch = args.arch or tr.get("arch") or DEFAULT_ARCH[task]
    cfg.setdefault("training", {})["arch"] = arch
    model = build_model(task, len(classes), not args.no_pretrained, arch).to(dev)
    print(f"model {arch}: {count_params(model)/1e6:.1f}M parameters on {dev}", flush=True)
    w0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=tr.get("lr", 3e-4), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cw = None
    if task == "classification" and tr.get("class_weighting", True):
        cnt = np.bincount([int(r["label"]) for r in split["train"]], minlength=len(classes)).astype(float)
        cw = torch.tensor(np.where(cnt > 0, cnt.sum() / (len(classes) * np.maximum(cnt, 1)), 0), dtype=torch.float32, device=dev)

    val0, _ = summarise(predict(model, dl["val"], task, dev), split["val"], task, classes)
    hist, best, best_ep, t0 = [], -1.0, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y, _ in dl["train"]:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = seg_loss(logits, y, len(classes)) if task == "segmentation" else F.cross_entropy(logits, y, weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        vres, _ = summarise(predict(model, dl["val"], task, dev), split["val"], task, classes)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_score": vres["selection_score"],
                     "sec": round(time.time() - t0, 1)})
        print(f"epoch {ep}/{epochs}  loss {hist[-1]['train_loss']:.4f}  val {vres['selection_metric']} {vres['selection_score']:.4f}", flush=True)
        if vres["selection_score"] > best:
            best, best_ep = vres["selection_score"], ep
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": ep, "input_size": size,
                        "pretrained_encoder": not args.no_pretrained}, out / "checkpoint_best.pt")
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader(), w.writerows(hist)

    # reload best checkpoint, verify, then (only now) test
    ck = torch.load(out / "checkpoint_best.pt", map_location=dev)
    model.load_state_dict(ck["state_dict"])
    changed = any(not torch.equal(w0[k].to(dev), v) for k, v in model.state_dict().items() if v.dtype.is_floating_point)
    vbest, _ = summarise(predict(model, dl["val"], task, dev), split["val"], task, classes)
    metrics = {"task_type": task, "class_names": classes, "arch": arch, "device": str(dev), "torch": torch.__version__,
               "pretrained_encoder": not args.no_pretrained, "epochs": epochs, "selected_epoch": best_ep,
               "selection_metric": vbest["selection_metric"], "val_before_training": val0["selection_score"],
               "val_selected": vbest["selection_score"], "weights_changed": bool(changed),
               "checkpoint_reload_verified": abs(vbest["selection_score"] - best) < 1e-6,
               "n_frames": {s: len(v) for s, v in split.items()},
               "n_patients": {s: len({r['patient_id'] for r in v}) for s, v in split.items()},
               "train_seconds": round(time.time() - t0, 1), "smoke_test": bool(args.limit or args.no_pretrained)}
    if split["test"]:
        preds = predict(model, dl["test"], task, dev)
        tres, per_case = summarise(preds, split["test"], task, classes)
        metrics["test"] = tres
        with open(out / "per_case_test.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_case[0]))
            w.writeheader(), w.writerows(per_case)
        save_examples(preds, split["test"], dl["test"].dataset, classes, out / "examples", task)
    else:
        metrics["test"] = None
        metrics["note"] = "no test split — results are validation only"
    write_json(metrics, out / "metrics.json")
    print(json.dumps({k: metrics[k] for k in ("selected_epoch", "val_before_training", "val_selected",
                                              "weights_changed", "checkpoint_reload_verified")}, indent=2))


if __name__ == "__main__":
    main()
