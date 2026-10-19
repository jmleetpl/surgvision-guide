@echo off
REM SurgVision Studio 실행 (Windows). 데이터는 이 컴퓨터에만 저장됩니다.
cd /d "%~dp0"
where ffmpeg >nul 2>nul || echo [경고] ffmpeg가 없습니다. 설치: conda install -c conda-forge ffmpeg
python -c "import torch, torchvision, streamlit" 2>nul || (echo torch/torchvision/streamlit 이 없습니다. 실행: pip install -r requirements.txt & pause & exit /b 1)
python -m streamlit run app.py
pause
