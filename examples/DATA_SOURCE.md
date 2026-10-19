# 예제 데이터 출처와 변경 내역

Hong W-Y, Kao C-L, Kuo Y-H, Wang J-R, Chang W-L, Shih C-S. **CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80.** 2020. https://arxiv.org/abs/2012.12453 · 공식 배포 https://www.kaggle.com/datasets/newslab/cholecseg8k

- 라이선스: **CC BY-NC-SA 4.0** (출처 표기·비상업·동일조건변경허락). 원저자의 보증이나 승인을 뜻하지 않습니다.
- 변경 내역(2026-10-17):
  1. 오남기 강사 실습 자료(2026-10-11)의 432쌍 부분집합(video01·09·12·18·20, 256×144, 간=1 / 그 외=0 / 미정의=255)을 사용했습니다.
  2. 영상별 프레임을 1 fps 동영상(`example_videos/hosp_case_1~5.mp4`)으로 재구성해 "병원 영상"처럼 쓰도록 했습니다. 원본 수술 영상이 아니며 프레임 간 시간 간격은 원본과 다릅니다.
  3. `example_reference_labels.zip`은 위 간 마스크를 CVAT Segmentation Mask 1.1 형식(liver = 250,50,83)으로 바꾼 것입니다. 미정의(255) 영역은 배경으로 들어갑니다. 독립 전문가가 새로 판정한 임상 정답이 아닙니다.
  4. `example_videos/linkage.csv`의 환자 번호(P01~P05)는 영상 번호와 1:1인 가상 번호입니다.
- 재배포 시 이 파일을 함께 제공하고 비상업 조건을 유지하세요.
