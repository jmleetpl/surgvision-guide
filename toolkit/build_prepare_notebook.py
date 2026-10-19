"""Generate prepare_data.ipynb — runs INSIDE the hospital (local Jupyter).

Raw videos and the linkage file never leave the machine running this notebook.
Steps: extract+de-identify -> human out-of-body review -> select frames into
CVAT batches -> (annotate in CVAT) -> import exports -> patient split -> bundle.

Usage:
  python build_prepare_notebook.py --pack packs/<slug>/pack.json --out packs/<slug>/prepare_data.ipynb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_notebook import code, md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = json.loads(Path(args.pack).read_text(encoding="utf-8"))
    d, a = p["data"], p["annotation"]
    mode = a["mode"]
    import_cell = {
        "seg": "cmd += ['seg', '--manifest', 'annotate/annotation_manifest.csv', '--export', EXPORT, '--frames', 'frames']",
        "tag": "cmd += ['tag', '--manifest', 'annotate/annotation_manifest.csv', '--export', EXPORT, '--frames', 'frames']",
        "interval": "cmd += ['interval', '--manifest', 'frames/frames_manifest.csv', '--intervals', EXPORT, '--frames', 'frames']",
    }[mode]
    export_hint = {"seg": "CVAT Segmentation Mask 1.1 ZIP (영상별 Task마다 하나)",
                   "tag": "CVAT for images 1.1 ZIP (영상별 Task마다 하나)",
                   "interval": "phase 구간 CSV (video_id,start_sec,end_sec,label)"}[mode]

    cells = [
        md(f"""
# {p['title_ko']} — 원내 데이터 준비 노트북

**이 노트북은 병원 안 컴퓨터의 로컬 Jupyter에서만 실행합니다.** 원본 영상과 `linkage.csv`(원래 파일명↔연구번호 대응표)는 이 컴퓨터 밖으로 내보내지 않습니다.

| 단계 | 하는 일 | 사람이 할 일 |
|---|---|---|
| A | 프레임 추출·메타데이터 제거·화면 글자 가리기 | 원본 영상 폴더와 linkage.csv 준비 |
| B | 체내 밖(out-of-body) 화면 검토 | 접촉 시트를 보고 제외할 프레임 지정 |
| C | 아노테이션할 프레임 고르기 → CVAT 배치 ZIP | — |
| D | (CVAT에서 아노테이션) | 라벨 정의서대로 직접 표시 |
| E | CVAT 결과 가져오기 | export 파일 위치 지정 |
| F | 환자 단위 분할 → 학습용 묶음 만들기 | 결과 확인 |
"""),
        code(f"""
import os, sys, json, csv, subprocess, pathlib
TK = pathlib.Path('toolkit')          # 이 팩의 toolkit 폴더
RAW_VIDEOS = 'raw_videos'             # 원본 영상 폴더 (병원 PC 안)
LINKAGE = 'linkage.csv'               # original_file,patient_id,video_id
FPS = {d['extract_fps']}
MODALITY = '{d.get('modality', 'laparoscopic')}'   # microscopic/open → 자동 체내밖 후보 표시 끔, 전부 사람 검토
BLACKOUT = '{d.get('blackout', '')}'   # 화면에 이름·날짜가 박혀 있으면 "x,y,w,h" (원본 픽셀)
for f in (RAW_VIDEOS, LINKAGE):
    print(f, '있음' if os.path.exists(f) else '❌ 없음')
gov = json.loads(pathlib.Path('governance_decision.json').read_text())
print('거버넌스 결정:', json.dumps(gov, ensure_ascii=False, indent=1))
assert gov.get('irb_status') in ('approved', 'exempt_confirmed', 'public_data'), '[STOP] IRB 승인(또는 면제 확인) 전에는 환자 영상을 처리하지 않습니다.'
"""),
        md("## A. 프레임 추출과 비식별 처리\n\n메타데이터는 지우고, 파일명은 연구번호로 바꿉니다. 화면 구석에 이름·등록번호가 박혀 있으면 위 `BLACKOUT`에 좌표를 넣으세요."),
        code("""
cmd = [sys.executable, str(TK/'extract_frames.py'), '--videos', RAW_VIDEOS, '--linkage', LINKAGE, '--out', 'frames', '--fps', str(FPS),
       '--modality', MODALITY]
if BLACKOUT: cmd += ['--blackout', BLACKOUT]
subprocess.run(cmd, check=True)
"""),
        md("""
## B. 체내 밖 화면 검토 — 사람이 반드시 확인

색 기반 자동 표시(`OOB?`)는 **후보일 뿐**입니다. 접촉 시트를 모두 보고, 수술실·얼굴·모니터·서류 등 체내가 아닌 화면이나 식별 가능한 정보가 보이는 프레임 번호를 `EXCLUDE`에 적습니다.
자동 표시가 틀렸으면 `KEEP`에 적어 되살립니다.
"""),
        code("""
from IPython.display import Image as IImage, display
for p in sorted(pathlib.Path('frames/review').glob('contact_*.jpg')):
    print(p.name); display(IImage(str(p)))
"""),
        code("""
EXCLUDE = []   # 예: ['V01_000123', 'V01_000124']
KEEP = []      # 자동 OOB 표시를 취소할 프레임
REVIEWER = ''  # 검토자 이니셜
assert REVIEWER, '검토자 이니셜을 적어 주세요'
rows = list(csv.DictReader(open('frames/frames_manifest.csv', encoding='utf-8')))
for r in rows:
    if r['sample_id'] in EXCLUDE: r['oob_flag'] = '1'
    if r['sample_id'] in KEEP: r['oob_flag'] = '0'
    r['oob_reviewed'] = '1'; r['oob_reviewer'] = REVIEWER
with open('frames/frames_manifest.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('제외', sum(r['oob_flag']=='1' for r in rows), '/', len(rows), '프레임')
"""),
        md(f"## C. 아노테이션할 프레임 고르기\n\n영상마다 최대 {a['frames_per_video']}장, 최소 {a['min_gap_sec']}초 간격으로 고릅니다. 영상별 ZIP 하나 = CVAT Task 하나입니다."),
        code(f"""
subprocess.run([sys.executable, str(TK/'select_frames.py'), '--frames', 'frames', '--per-video', '{a['frames_per_video']}',
                '--min-gap-sec', '{a['min_gap_sec']}', '--out', 'annotate'], check=True)
""" if mode != "interval" else """
print('구간(interval) 방식은 프레임 선택 없이 전체 추출 프레임에 phase 구간을 붙입니다. CVAT 대신 구간 시트를 작성하세요.')
"""),
        md(f"""
## D. CVAT 아노테이션 (노트북 밖)

`ASIDE_S4_ANNOTATE.txt` 안내대로 진행합니다. 끝나면 결과 파일을 `exports/` 폴더에 둡니다: **{export_hint}**
"""),
        md("## E. CVAT 결과 가져오기\n\n아노테이션 표시가 없는 프레임은 '미완료'로 남고 학습에서 빠집니다. 빈 마스크를 음성 정답으로 쓰지 않습니다."),
        code(f"""
import glob, shutil
exports = sorted(glob.glob('exports/*'))
print('export 파일:', exports)
merged = None
for i, EXPORT in enumerate(exports):
    cmd = [sys.executable, str(TK/'cvat_import.py')]
    {import_cell}
    if merged: cmd[cmd.index('--manifest') + 1] = merged
    cmd += ['--config', 'config.json', '--out', f'labeled_{{i}}']
    if {str(a.get('reviewed_empty_file') is not None)}: cmd += ['--reviewed-empty', 'reviewed_empty.txt']
    subprocess.run(cmd, check=True)
    merged = f'labeled_{{i}}/labeled_manifest.csv'   # 다음 export는 이 manifest에 이어 붙임
print('최종 manifest:', merged)
"""),
        md("## F. 환자 단위 분할과 학습 묶음\n\n같은 환자가 학습·검증·시험에 섞이면 STOP으로 멈춥니다."),
        code(f"""
subprocess.run([sys.executable, str(TK/'make_splits.py'), '--manifest', merged, '--ratios', '{p['split']['ratios']}',
                '--seed', '{p['split']['seed']}', '--out', 'split'] + ({json.dumps(['--test-patients', ','.join(p['split'].get('test_patients', []))]) if p['split'].get('test_patients') else '[]'}), check=True)
subprocess.run([sys.executable, str(TK/'bundle.py'), '--config', 'config.json', '--split', 'split',
                '--governance', 'governance_decision.json', '--out', 'study_bundle.zip'], check=True)
print(open('split/split_summary.json', encoding='utf-8').read())
"""),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    Path(args.out).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("notebook ->", args.out)


if __name__ == "__main__":
    main()
