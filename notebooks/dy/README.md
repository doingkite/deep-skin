# dy: Forehead/Glabella ResNet-50 Multitask Pipeline

이 폴더는 이마와 미간 파트 전용 작업물입니다. 흐름은 `crop.py`로 필요한 부위 이미지를 만들고, `train.py`로 ResNet-50 공유 백본 기반 멀티태스크 모델을 학습한 뒤, `predict.py`로 등급·회귀 수치·추천 근거·Grad-CAM을 생성하는 구조입니다.

## 0. Python Environment

아래 명령은 로컬 Windows 환경에 `.venv`와 PyTorch CUDA가 설치된 경우의 예시입니다. 저장소를 새로 복제한 컴퓨터에는 가상환경과 원본 데이터가 포함되지 않습니다.

저장소 루트에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import torch, torchvision, pandas; print(torch.__version__, torchvision.__version__, pandas.__version__)"
.\.venv\Scripts\python.exe -c "import torch; print(torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

새 터미널에서 `uv`가 바로 잡히지 않으면 터미널을 다시 열거나 아래처럼 현재 세션 PATH만 갱신합니다.

```powershell
$env:PATH = "C:\Users\WOO\.local\bin;$env:PATH"
uv --version
```

## 1. Crop + Metadata

```powershell
cd C:\Users\WOO\Documents\GitHub\deep-skin
.\.venv\Scripts\python.exe notebooks\dy\crop.py --data-root "C:\Users\WOO\Downloads\028.한국인 피부상태 측정 데이터" --save-samples 20
```

생성 결과:

- `data/cropped/dy_forehead_glabella/{train,val}/{forehead,glabella}`
- `data/processed/dy_forehead_glabella_train_metadata.csv`
- `data/processed/dy_forehead_glabella_val_metadata.csv`
- `results/dy_forehead_glabella/crop_summary.json`
- `results/dy_forehead_glabella/crop_samples`

기록된 전체 크롭 생성 결과는 train 8,580장, val 1,070장입니다. 실제 재실행 결과는 입력 데이터에 따라 달라질 수 있습니다.

기본 각도는 정면 계열 `F`, `Ft`, `Fb`입니다. 측면 각도까지 쓰려면 `--include-side-angles`를 추가합니다.

## 2. Multitask Training

먼저 학습/검증 CSV의 인물 ID와 원본 사진 경로가 겹치지 않는지 검사합니다. 두 학습 스크립트도 시작할 때 같은 검사를 수행하며, 겹치면 중단합니다.

```powershell
.\.venv\Scripts\python.exe notebooks\dy\data_checks.py `
  --train-csv data\processed\dy_forehead_glabella_train_metadata.csv `
  --val-csv data\processed\dy_forehead_glabella_val_metadata.csv
```

```powershell
.\.venv\Scripts\python.exe notebooks\dy\train.py `
  --epochs 40 --batch-size 16 --weights local --grade-scheme three
```

학습 태스크:

- `forehead_pigmentation`: 이마 색소침착 (원본 0~5, 위 명령의 3등급 설정에서는 0~2)
- `forehead_wrinkle`: 이마 주름 (원본 0~6, 3등급 설정에서는 0~2)
- `glabella_wrinkle`: 미간 주름 (원본 0~6, 3등급 설정에서는 0~2)
- 회귀: 이마 수분 및 주요 탄력 수치

체크포인트와 학습 로그는 `checkpoints/trained/dy_forehead_glabella/{run}` 아래에 저장됩니다.

기본 설정 `--weights local`은 `checkpoints/pretrained/resnet50-0676ba61.pth`가 없으면 즉시 실패합니다. 캐시나 네트워크 사용이 가능하면 `--weights torchvision`을 쓸 수 있습니다. 의도적인 무작위 초기화에는 `--weights none`을 쓰세요. 이전 설정 `local_or_none`은 호환용으로 남아 있지만, 파일이 없어 무작위로 시작하면 경고합니다. 실험 표에는 사용한 가중치 출처와 세부 설정을 남기세요.

### DINOv3 특징 기반 비교 실험

`train_dinov3.py`는 DINOv3의 대표 특징을 먼저 저장하고 백본을 고정한 채 분류·회귀 헤드를 학습합니다. Hugging Face 모델 접근 권한과 `HF_TOKEN`이 필요할 수 있습니다. 비교 시 ResNet과 같은 학습/검증 CSV 및 `--grade-scheme three`를 지정하고, 가능하면 `--model-revision`에 모델의 **커밋 해시**를 넣으세요.

```powershell
.\.venv\Scripts\python.exe notebooks\dy\train_dinov3.py `
  --grade-scheme three --model-revision COMMIT_SHA
```

실행할 때 `COMMIT_SHA`를 실제 모델 커밋 해시로 바꿉니다. 커밋 해시를 모르면 처음에는 `--model-revision COMMIT_SHA` 부분을 생략해도 됩니다.

특징 캐시는 CSV 내용, 사진 파일 크기·수정 시각, 전처리 설정과 모델 리비전이 일치할 때만 재사용합니다. 사진 내용만 바뀌고 파일 크기와 수정 시각이 모두 유지되었다면 `--force-embeddings`로 다시 추출하세요. 리비전을 확인할 수 없으면 캐시를 재사용하지 않습니다.

## 3. Inference + XAI + Recommendation

```powershell
.\.venv\Scripts\python.exe notebooks\dy\predict.py `
  --checkpoint checkpoints\trained\dy_forehead_glabella\{run}\best.pth `
  --image data\cropped\dy_forehead_glabella\val\forehead\sample.jpg `
  --part forehead `
  --output-json results\dy_forehead_glabella\sample_report.json `
  --gradcam-dir results\dy_forehead_glabella\gradcam
```

응답에는 백엔드 계약에 맞춘 `parts` 배열, 회귀 수치, 수치 기반 설명, 추천 카테고리/성분, Grad-CAM 이미지 경로가 포함됩니다.

## 4. Design Notes

- 단일 등급 모델 대신 하나의 ResNet-50 백본에서 분류 head 3개와 회귀 head 1개를 동시에 학습합니다.
- 크롭 CSV에는 raw JSON, annotation, equipment 값을 같이 보존해 추후 분석과 오류 추적이 쉽도록 했습니다.
- 회귀 수치는 설명 가능성을 위한 보조 근거입니다. 예를 들어 이마 수분 예측값이 낮으면 보습/장벽 성분 추천 근거로 사용합니다.
- Grad-CAM은 분류 등급별로 모델 출력에 민감한 영역을 살펴보는 보조 시각화입니다. 판단의 인과적 근거를 증명하지는 않습니다.
