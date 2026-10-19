#!/bin/bash
# SurgVision Studio 실행 (macOS: 더블클릭). 데이터는 이 컴퓨터에만 저장됩니다.
cd "$(dirname "$0")"
PY="${SURGVISION_PYTHON:-}"
if [ -z "$PY" ]; then
  for c in /opt/anaconda3/bin/python "$HOME/anaconda3/bin/python" "$HOME/miniconda3/bin/python" python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch, torchvision, streamlit" 2>/dev/null; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "torch·torchvision·streamlit이 있는 Python을 찾지 못했습니다."
  echo "설치: python3 -m pip install -r requirements.txt   (그리고 ffmpeg: brew install ffmpeg)"
  read -p "Enter를 누르면 닫힙니다"; exit 1
fi
command -v ffmpeg >/dev/null || echo "⚠ ffmpeg가 없습니다. 영상 추출 전에 설치하세요: brew install ffmpeg"
echo "Python: $PY"
"$PY" -m streamlit run app.py
