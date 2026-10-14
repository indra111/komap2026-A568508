# 팀서유기 (접수번호 A568508) — 고탄성 알루미늄 합금 이미지 상 분할 모델

KoMaP 2026 「고탄성 알루미늄 합금 이미지 상 분할 인공지능 모델」 제출 분석 프로그램이다.
현미경 사진에서 Al 기지 · 초정 Si · Al₃Ni · 공정 Si 네 조직을 픽셀 단위로 구분한다.

- 검증 성능(validation 20장, 이미지별 4-class mIoU 평균): **79.90%**
- 학습 데이터: 주최 측 train 70장만 사용 (validation은 모델 선택에만 사용)
- **화면 캡처가 포함된 사용 매뉴얼: [README.html](README.html)** (웹 화면·명령줄 단계별 안내)
- 저장소: <https://github.com/indra111/komap2026-A568508> · 가중치: [Release v1.0](https://github.com/indra111/komap2026-A568508/releases/tag/v1.0)

## 빠른 시작

```bash
conda env create -f environment.yml && conda activate komap-2026-team-seoyugi
./download_weights.sh                    # 학습된 가중치 10개(5.3GB)를 Release에서 받아 SHA256 검증
./run_predict.sh test outputs/test_masks # test 10장 → *_mask.png (제출 결과 재현)
python app.py                            # 웹 화면: http://localhost:7860 에서 사진을 올려 분할
```

NVIDIA GPU(24GB 권장, CUDA 12.8 드라이버)가 필요하다. RTX A5000 24GB에서 test 10장 약 10분, val 20장 약 20분.

## 데이터 배치

주최 측 `train.zip`, `valid.zip`, `test.zip`을 아래처럼 풀어 둔다(`data/manifest.json`이 이 경로를 가리킨다).
```
data/extracted/train/{images,masks}/
data/extracted/valid/{images,masks}/
data/extracted/test/images/
```

## 실행 방법

| 목적 | 명령 | 결과 |
|---|---|---|
| test 결과 재현 | `./run_predict.sh test outputs/test_masks` | 10장 마스크(정답과 같은 RGB 색상) |
| 검증 점수 재현 | `./run_predict.sh val outputs/val_masks` | 20장 마스크 + `prediction-manifest.json`의 mIoU |
| 임의 사진 폴더 | `python predict_opus_champion.py --recipe review/opus/recipe-lov16-2head-al3ni110-crfbw05-psi100.json --images <폴더> --work /tmp/w --output <출력>` | 폴더의 PNG/JPG/TIF 전부 |
| 웹 화면 | `python app.py` → <http://localhost:7860> | 아래 세 영역을 브라우저에서 실행 |

### 웹 화면 (`python app.py`)

`run_predict.sh`가 하는 일을 모두 브라우저에서 실행할 수 있다.

1. **준비 상태** — GPU, 가중치(10개), 주최 측 데이터(test 10 / val 20)를 확인하고, 가중치가 없으면 [가중치 받기]로 GitHub Release에서 받아 SHA256을 검증한다.
2. **제출 결과 재현** — [test 10장 마스크 만들기]는 제출 결과와 같은 마스크를, [val 20장 점수 확인]은 검증 mIoU와 이미지별 점수·정답 비교를 보여준다. 마스크는 zip으로 내려받는다.
3. **내 사진 분석** — 현미경 사진을 끌어다 놓으면 분할 결과, 겹쳐 보기, 조직별 면적 분율·입자 수를 보여준다.

출력 색상: Al 기지 (153,127,76) / 초정 Si (76,178,76) / Al₃Ni (25,76,153) / 공정 Si (204,204,204).
웹 화면은 추가 패키지 없이 Python 표준 HTTP 서버로 동작하며, 사진은 이 컴퓨터 밖으로 나가지 않는다(`app_jobs/`에 저장).
웹 화면과 `run_predict.sh`는 같은 추론 코드를 쓰며, 같은 이미지에 대해 바이트 단위로 같은 마스크를 낸다.

## 모델 구성 (`review/opus/recipe-lov16-2head-al3ni110-crfbw05-psi100.json`)

1. **배율 정규화:** 폭 800px 미만 이미지를 ×1.85 확대해 관측 배율을 맞추고, 전체를 ×1.5(HRNet 하나는 ×2.0)로 키워 추론한 뒤 원본 해상도로 되돌린다.
2. **teacher 8개 결합(각 D4 8방향 평균):** DINOv3 Large(EMA) 9.4% · DINOv2 Large(LLRD+Lovász 16ep) 15% · DINOv2 Large(30ep) 9.4% · HRNet-OCR ×1.5 3종 9.4%씩 · HRNet-OCR ×2.0 18.9% · DINOv2 Base(HE50) 18.9%
3. **잔차 보정기 2개 평균**(D4) → **Al₃Ni 확률 ×1.1** → **DenseCRF**(양방향 가중치 0.5) → **RankIoU** → 100px 미만 초정 Si 조각을 공정 Si로 재분류

## 파일 구성

| 경로 | 내용 |
|---|---|
| `run_predict.sh` | **실행 파일.** test/val 마스크 생성 |
| `app.py` | 로컬 웹 인터페이스 |
| `download_weights.sh`, `weights-manifest.json` | 가중치 다운로드·SHA256 검증 |
| `predict_opus_champion.py` | 최종 추론기(teacher → 보정기 → 후처리) |
| `review/opus/recipe-*.json` | 최종 모델 구성(가중치 경로·비중·후처리 설정) |
| `scale_norm.py`, `opus_models.py` | 배율 정규화, 구성원 모델 생성·불러오기 |
| `train_scale_norm_large.py` | 구성원 학습기(`--family large/base/hrnet/dinov3`, `--ema`, `--llrd`, `--lovasz`, `--init`, `--epochs`, `--seed`) |
| `train_opus_refiner.py`, `train_focused_refiner_warm.py` 등 | 잔차 보정기 학습 |
| `eval_teacher_mix.py` | 구성원 캐시·결합·분율 보정 평가 |
| `environment.yml` | 개발 환경 (Python 3.12.3, PyTorch 2.10 CUDA 12.8) |
| `.crf-deps/` | 검증된 pydensecrf 1.0 빌드 (Linux x86_64, CPython 3.12) |

## 구성원 재학습 예

```bash
python train_scale_norm_large.py --family dinov3 --base-scale 1.5 --ema 0.999 --run runs/opus-scale-norm-dinov3-x1p5-ema
python train_scale_norm_large.py --family large --base-scale 1.5 --seed 20261016 --ema 0.999 --llrd 0.9 --llrd-lr 6e-5 --run runs/opus-scale-norm-large-x1p5-ema-llrd
python train_scale_norm_large.py --family large --base-scale 1.5 --seed 20261092 --init runs/opus-scale-norm-large-x1p5-ema-llrd/best.pth \
  --lovasz 0.5 --ema 0.999 --llrd 0.9 --llrd-lr 1e-5 --head-lr 1e-4 --epochs 16 --run runs/opus-scale-norm-large-x1p5-llrd-lovasz-ft16
python train_scale_norm_large.py --family large --base-scale 1.5 --epochs 30 --seed 20261013 --run runs/opus-scale-norm-large-x1p5-e30
python train_scale_norm_large.py --family hrnet --base-scale 1.5 --run runs/opus-scale-norm-hrnet-x1p5-v1
python train_scale_norm_large.py --family hrnet --base-scale 1.5 --seed 20261012 --run runs/opus-scale-norm-hrnet-x1p5-s2
python train_scale_norm_large.py --family hrnet --base-scale 1.5 --seed 20261042 --epochs 30 --run runs/opus-scale-norm-hrnet-x1p5-e30-s5
python train_scale_norm_large.py --family hrnet --base-scale 2.0 --seed 20261043 --run runs/opus-scale-norm-hrnet-x2p0-s6
```
DINOv2·DINOv3·HRNet 공개 사전학습 가중치는 학습 시작 시에만 필요하다(`data/pretrained/*`). 추론은 Release의 `best.pth`만 사용한다.
모든 학습은 train 70장만 사용했으며 validation 20장은 모델 선택에만 사용했다.
