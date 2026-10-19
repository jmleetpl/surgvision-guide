"""SurgVision Studio — project state, tool runner and rule-based recommendations.

The app never re-implements analysis: every computation is a toolkit script run
as a subprocess, so the app, the Colab notebooks and the Claude Code harness
produce identical files.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TK = ROOT / "toolkit"
PROJECTS = Path(os.environ.get("SURGVISION_PROJECTS", Path.home() / "SurgVisionProjects"))
PY = sys.executable
sys.path.insert(0, str(TK))

STEPS = [
    ("home", "0. 시작"),
    ("design", "1. 아이디어 → 설계"),
    ("size", "2. 필요한 건수"),
    ("gov", "3. 거버넌스·비식별"),
    ("data", "4. 데이터 넣기"),
    ("review", "5. 화면 검토(비식별)"),
    ("annotate", "6. 아노테이션"),
    ("split", "7. 데이터 구성"),
    ("search", "8. 모델 후보 비교"),
    ("train", "9. 학습·평가"),
    ("external", "10. 외부검증"),
    ("paper", "11. 결과·원고"),
]

MODALITIES = {"laparoscopic": "복강경", "robotic": "로봇", "endoscopic": "내시경(관강 내/비강 등)",
              "microscopic": "현미경(백내장·미세수술)", "open": "개복/개방 수술(헤드캠·천장 카메라)"}


# ------------------------------------------------------------------ projects
def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s or "study_" + dt.datetime.now().strftime("%Y%m%d_%H%M")


def list_projects() -> list[str]:
    PROJECTS.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in PROJECTS.iterdir() if (p / "pack.json").exists())


def pdir(slug: str) -> Path:
    return PROJECTS / slug


def rj(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def wj(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_config(proj: Path, pack: dict) -> None:
    wj(proj / "config.json", {"task_type": pack["task_type"], "class_names": pack["class_names"],
                              "training": pack["training"]})


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(), w.writerows(rows)


def status(proj: Path) -> dict[str, bool]:
    """What exists on disk — the progress checklist is derived, never stored."""
    lab = proj / "labeled" / "labeled_manifest.csv"
    n_done = sum(1 for r in read_csv(lab) if r.get("annotation_status") == "done") if lab.exists() else 0
    runs = sorted((proj / "runs").glob("*/metrics.json")) if (proj / "runs").exists() else []
    gov = rj(proj / "governance_decision.json", {})
    fm = proj / "frames" / "frames_manifest.csv"
    reviewed = fm.exists() and all(r.get("oob_reviewed") == "1" for r in read_csv(fm))
    return {"design": (proj / "pack.json").exists(),
            "size": (proj / "sample_size" / "sample_size_precision.json").exists(),
            "gov": bool(gov.get("decided_by")),
            "data": fm.exists() or (proj / "labeled" / "import_report.json").exists(),
            "review": reviewed or (proj / "labeled" / "import_report.json").exists() and not fm.exists(),
            "annotate": n_done > 0,
            "split": (proj / "split" / "split_summary.json").exists(),
            "search": (proj / "search" / "search_summary.json").exists(),
            "train": bool(runs),
            "external": bool(list((proj / "external").glob("*/external_metrics.json"))) if (proj / "external").exists() else False,
            "paper": (proj / "manuscript_draft.md").exists()}


def latest_run(proj: Path) -> Path | None:
    runs = sorted((proj / "runs").glob("*/metrics.json")) if (proj / "runs").exists() else []
    return runs[-1].parent if runs else None


# ------------------------------------------------------------------ runner
def run_tool(script: str, args: list[str], cwd: Path, log_box=None) -> tuple[int, str]:
    """Run a toolkit script, streaming output into a Streamlit placeholder."""
    cmd = [PY, "-u", str(TK / script)] + [str(a) for a in args]
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        if log_box is not None:
            log_box.code("\n".join(lines[-40:]) or "…", language=None)
    proc.wait()
    text = "\n".join(lines)
    (cwd / "logs").mkdir(exist_ok=True)
    (cwd / "logs" / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{script.replace('.py', '')}.log").write_text(
        " ".join(cmd) + "\n\n" + text, encoding="utf-8")
    return proc.returncode, text


def gov_allows_data(proj: Path) -> tuple[bool, str]:
    g = rj(proj / "governance_decision.json", {})
    if not g.get("decided_by"):
        return False, "3단계 거버넌스를 먼저 확정하세요."
    if g.get("irb_status") not in ("approved", "exempt_confirmed", "public_data"):
        return False, "IRB 승인(또는 면제 확인) 전에는 환자 영상을 처리하지 않습니다. 공개 데이터로 연습하세요."
    return True, ""


# ------------------------------------------------------------------ recommendations
def recommend(task: str, mode: str, absent_common: bool, small_structure: bool, rare_important: bool) -> dict:
    """Rule-based defaults distilled from Metrics Reloaded (Maier-Hein & Reinke, Nat Methods 2024)."""
    if task == "segmentation":
        primary = "per-image Dice, averaged over test images, patient-level bootstrap 95% CI"
        sec = ["IoU", "median Dice"]
        why = ["영역이 얼마나 겹치는지(Dice)가 분할의 기본 지표입니다."]
        if absent_common:
            sec += ["images with structure missed entirely", "false-alarm images"]
            why.append("구조가 없는 프레임이 흔하므로, 평균 Dice가 가리는 '통째로 놓침/없는데 그림' 이미지 수를 따로 셉니다.")
        if small_structure:
            why.append("작은 구조는 픽셀 몇 개 차이로 Dice가 크게 흔들립니다. 경계 거리 지표(NSD)는 v1 미구현이므로 best/worst 그림 검토로 보완하고 한계에 적습니다.")
        ann = ("CVAT Brush/Polygon 마스크. 처음 2장은 손으로, 이후 SAM 보조 가능. 서로 떨어진 프레임이므로 보간·추적 금지.")
        cands = ["unet_resnet18", "lraspp_mobilenet", "deeplabv3_mobilenet"]
    else:
        if mode == "interval":
            primary = "macro F1 over test frames, with per-video F1 distribution"
            ann = "영상을 보며 phase 시작·끝 시각을 구간 표에 기록(앱 안 표 편집)."
            why = ["phase는 시간 구간이므로 구간 표가 프레임마다 태그하는 것보다 훨씬 빠릅니다.",
                   "프레임 단위 분류는 앞뒤 맥락을 쓰지 않습니다(한계로 기록)."]
        else:
            primary = "balanced accuracy, patient-level bootstrap 95% CI"
            ann = "앱 안 태거로 프레임마다 클래스 하나(판단 보류는 제외)."
            why = ["클래스 비율이 치우치기 쉬워 정확도 대신 클래스별 recall 평균(balanced accuracy)을 씁니다."]
        sec = ["macro F1", "per-class recall and precision", "per-class AUROC (one-vs-rest)", "confusion matrix"]
        if rare_important:
            why.append("드물지만 놓치면 안 되는 클래스가 있으면 그 클래스의 recall(민감도)을 공동 1차 지표로 보고합니다.")
        cands = ["resnet18", "efficientnet_b0", "mobilenet_v3_large"]
    return {"primary": primary, "secondary": sec, "rationale": " ".join(why), "annotation": ann, "candidates": cands}


def new_pack(**kw) -> dict:
    task, mode = kw["task_type"], kw["mode"]
    rec = recommend(task, mode, kw.get("absent_common", False), kw.get("small_structure", False),
                    kw.get("rare_important", False))
    classes = (["background"] if task == "segmentation" else []) + kw["classes"]
    return {
        "slug": kw["slug"], "title_ko": kw["title_ko"], "title_en": kw.get("title_en") or kw["title_ko"],
        "research_question": kw["question"], "clinical_rationale": kw.get("rationale", ""),
        "task_type": task, "class_names": classes, "label_definitions": kw["definitions"],
        "hard_cases": kw["hard_cases"],
        "data": {"source": kw["source"], "procedure": kw["procedure"], "modality": kw["modality"],
                 "unit_hierarchy": "patient > video > frame", "n_patients_planned": kw["n_patients"],
                 "inclusion": kw.get("inclusion", ""), "exclusion": kw.get("exclusion", ""),
                 "extract_fps": kw["fps"], "blackout": ""},
        "annotation": {"mode": mode, "platform": "cvat_cloud" if mode == "seg" else ("in_app" if mode == "tag" else "spreadsheet"),
                       "platform_url": "https://app.cvat.ai/", "frames_per_video": kw["frames_per_video"],
                       "min_gap_sec": kw["min_gap"], "annotators": kw["annotators"],
                       "double_annotation_fraction": kw.get("double_fraction", 0.0), "reviewed_empty_file": None},
        "split": {"unit": "patient_id", "ratios": kw.get("ratios", "0.6,0.2,0.2"), "seed": 42,
                  "test_patients": kw.get("test_patients", [])},
        "training": {"epochs": kw.get("epochs", 10), "input_size": [320, 192] if task == "segmentation" else [224, 128],
                     "batch_size": 8, "lr": 0.0003, "seed": 42, "num_workers": 2, "arch": rec["candidates"][0]},
        "metrics": {"primary": rec["primary"], "secondary": rec["secondary"], "rationale": rec["rationale"],
                    "candidates": rec["candidates"]},
        "compute_platform": "local_jupyter",
        "reporting_guideline": "TRIPOD+AI (BMJ 2024) / CLAIM 2024",
        "governance": {"irb_status": "public_data" if kw["source"] == "public" else "pending",
                       "data_class": "public_dataset" if kw["source"] == "public" else "deidentified_frames"},
        "created": dt.date.today().isoformat(),
    }


def apply_tags(proj: Path, annotator: str, classes: list[str]) -> dict:
    """In-app tags -> labeled manifest (participant labels override earlier ones for the same frames)."""
    tags = proj / "labels" / f"tags_{annotator}.csv"
    if not tags.exists():
        return {"annotated": 0}
    t = {r["sample_id"]: r["label"] for r in read_csv(tags)}
    lab = proj / "labeled" / "labeled_manifest.csv"
    rows = read_csv(lab) if lab.exists() else read_csv(proj / "frames" / "frames_manifest.csv")
    fr = proj / "frames"
    n = 0
    for r in rows:
        if not Path(r["image_path"]).is_absolute():
            r["image_path"] = str((fr / r["image_path"]).resolve())
        if r.get("oob_flag") == "1":
            r["annotation_status"] = "excluded_oob"
            continue
        v = t.get(r["sample_id"])
        if v in classes:
            r["label"], r["annotation_status"], r["label_source"] = str(classes.index(v)), "done", annotator
            n += 1
    write_csv(rows, lab)
    return {"annotated": n, "uncertain": sum(1 for v in t.values() if v == "__uncertain__")}
