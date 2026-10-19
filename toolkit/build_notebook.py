"""Generate the participant notebook (Colab or local Jupyter) for one study pack.

The notebook only CALLS the toolkit scripts; all logic stays in the scripts so
the notebook, the Aside prompt and the manuscript describe the same code.

Usage:
  python build_notebook.py --config config.json --title "담낭절제 CVS 분류" --out pack/train_eval.ipynb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task = cfg["task_type"]
    video_hint = "overlay.mp4" if task == "segmentation" else "timeline.png"

    cells = [
        md(f"""
# {args.title} — 학습·평가 노트북

이 노트북은 툴킷 스크립트를 순서대로 부르기만 합니다. 각 단계 제목의 번호를 따라 **한 셀씩** 실행하세요. `Run all`은 쓰지 않습니다.

| 단계 | 하는 일 |
|---|---|
| 1 | 자료 묶음(`study_bundle.zip`) 열기 |
| 2 | 실행 환경 확인 |
| 3 | 데이터 분할 확인 (환자 단위·누수 검사) |
| 4 | 학습 (검증 자료로 epoch 선택) |
| 5 | 시험 결과·그림 확인 |
| 6 | 영상 추론 |
| 7 | 결과 보관 |

과제 유형: **{task}** · 클래스: {', '.join(cfg['class_names'])}
"""),
        md("""
## 1. 자료 묶음 열기

`study_bundle.zip` 안에는 `toolkit/`, `config.json`, `split/split_manifest.csv`, 이미지·마스크가 들어 있습니다.
**자체 환자 영상이면 거버넌스 결정서(`governance_decision.json`)에서 이 실행 장소가 허용됐는지 먼저 확인하세요.**
Colab에서는 업로드 창이 뜨고, 로컬 Jupyter에서는 같은 폴더의 ZIP을 바로 사용합니다.
"""),
        code("""
import os, sys, zipfile, json, time, pathlib
IN_COLAB = 'google.colab' in sys.modules
BUNDLE = 'study_bundle.zip'
if IN_COLAB and not os.path.exists(BUNDLE):
    from google.colab import files
    up = files.upload()
    assert BUNDLE in up, f'업로드한 파일 이름이 {BUNDLE} 이 아닙니다: {list(up)}'
assert os.path.exists(BUNDLE), f'{BUNDLE} 을 찾지 못했습니다. 노트북과 같은 폴더에 두세요.'
WORK = pathlib.Path('study')
if not (WORK / 'config.json').exists():
    zipfile.ZipFile(BUNDLE).extractall(WORK)
SESSION = time.strftime('%Y%m%d_%H%M%S')
RUN = WORK / 'runs' / SESSION
print('자료 폴더:', WORK.resolve()); print('이번 실행 결과 폴더:', RUN.resolve())
gov = WORK / 'governance_decision.json'
if gov.exists():
    g = json.loads(gov.read_text())
    print('거버넌스:', g.get('compute_platform'), '/', g.get('data_class'))
    if IN_COLAB and g.get('compute_platform') == 'local_only':
        raise SystemExit('[STOP] 이 연구 자료는 원내 로컬 실행만 허용됐습니다. Colab에서 실행하지 마세요.')
"""),
        md("## 2. 실행 환경 확인\n\n`device`가 `cuda`면 GPU, `cpu`면 CPU입니다. CPU도 지원 경로입니다."),
        code("""
import importlib, subprocess
for pkg in ('torch', 'torchvision', 'matplotlib', 'PIL'):
    importlib.import_module(pkg)
import torch
print('python', sys.version.split()[0], '| torch', torch.__version__)
print('device:', 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
if torch.cuda.is_available(): print('gpu:', torch.cuda.get_device_name(0))
print('ffmpeg:', subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.split('\\n')[0])
"""),
        md("""
## 3. 데이터 분할 확인

환자 단위로 나뉘었는지, 같은 환자가 두 집합에 들어가지 않았는지(`leakage: none`) 확인합니다.
`annotation_status`가 `done`인 프레임만 학습·평가에 들어갑니다. 아노테이션하지 않은 프레임은 음성 정답이 아닙니다.
"""),
        code("""
summary = json.loads((WORK / 'split' / 'split_summary.json').read_text())
print(json.dumps({k: summary[k] for k in ('unit', 'n_patients', 'n_frames', 'leakage', 'warnings')}, ensure_ascii=False, indent=1))
cfg = json.loads((WORK / 'config.json').read_text()); print('config:', json.dumps(cfg, ensure_ascii=False))
"""),
        md("""
## 4. 학습

검증 자료로 가장 좋은 epoch를 고르고, 그 뒤에 **한 번만** 시험 자료로 평가합니다.
시험 결과를 본 뒤 설정을 바꿔 다시 돌리면 시험 자료가 검증 자료가 되어 버립니다. 그 경우 원고에 그렇게 적어야 합니다.
실행 중에는 같은 셀을 다시 누르지 마세요.
"""),
        code("""
cmd = [sys.executable, str(WORK/'toolkit'/'train.py'), '--config', str(WORK/'config.json'),
       '--manifest', str(WORK/'split'/'split_manifest.csv'), '--base', str(WORK), '--out', str(RUN)]
print(' '.join(cmd)); t0 = time.time()
r = subprocess.run(cmd); print('종료코드', r.returncode, '| 소요', round(time.time()-t0), '초')
assert r.returncode == 0 and (RUN/'checkpoint_best.pt').exists(), '학습 실패 — 위 오류의 마지막 줄을 기록하세요'
"""),
        md("""
## 5. 시험 결과와 그림

`weights_changed`와 `checkpoint_reload_verified`가 모두 True인지 먼저 봅니다.
표의 † 표시는 시험 환자 5명 미만으로 신뢰구간을 믿을 수 없다는 뜻입니다.
"""),
        code("""
r = subprocess.run([sys.executable, str(WORK/'toolkit'/'make_report.py'), '--split', str(WORK/'split'/'split_summary.json'),
                    '--run', str(RUN), '--out', str(RUN/'results')])
from IPython.display import Image as IImage, display, Markdown
m = json.loads((RUN/'metrics.json').read_text())
print({k: m[k] for k in ('selected_epoch', 'val_before_training', 'val_selected', 'weights_changed', 'checkpoint_reload_verified', 'device', 'train_seconds')})
for p in ('fig_learning_curve.png', 'fig_per_case.png'):
    if (RUN/'results'/p).exists(): display(IImage(str(RUN/'results'/p)))
for p in sorted((RUN/'examples').glob('*.png'))[:6]:
    print(p.name); display(IImage(str(p)))
"""),
        md(f"""
## 6. 영상 추론

`INFER_VIDEO`에 학습·검증에 쓰지 않은 영상 경로를 넣습니다. 시험 환자의 영상이면 시험 결과와 같은 근거이지, 추가 독립 평가가 아닙니다.
결과: `{video_hint}`
"""),
        code("""
INFER_VIDEO = ''   # 예: 'study/demo_video.mp4'  — 비워두면 묶음 안의 demo_video.mp4 사용
vid = INFER_VIDEO or str(WORK/'demo_video.mp4')
if os.path.exists(vid):
    subprocess.run([sys.executable, str(WORK/'toolkit'/'infer_video.py'), '--checkpoint', str(RUN/'checkpoint_best.pt'),
                    '--video', vid, '--fps', '2', '--out', str(RUN/'video')], check=True)
    if (RUN/'video'/'timeline.png').exists(): display(IImage(str(RUN/'video'/'timeline.png')))
    print('결과 폴더:', RUN/'video')
else:
    print('추론할 영상이 없습니다. INFER_VIDEO에 경로를 넣으세요.')
"""),
        md("## 7. 결과 보관\n\n`DOWNLOAD_RESULTS = True`로 바꾼 뒤 이 셀만 실행합니다. 노트북 저장과 결과 ZIP 저장은 별개입니다."),
        code("""
DOWNLOAD_RESULTS = False
if DOWNLOAD_RESULTS:
    import shutil
    z = shutil.make_archive(f'results_{SESSION}', 'zip', RUN)
    print('만든 파일:', z)
    if IN_COLAB:
        from google.colab import files; files.download(z)
"""),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("notebook ->", args.out)


if __name__ == "__main__":
    main()
