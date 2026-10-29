# MIR Chord Estimator

## 웹 서비스와 Render 배포

`src/chord_estimator/webapp`에 업로드, 작업 상태 조회, 타임라인 재생, JSON/CSV/LAB/원시 프레임 다운로드를 제공하는 FastAPI 웹 UI가 포함되어 있다. 모델은 서버 시작 시 한 번만 로딩되며 체크포인트가 없거나 검증에 실패하면 서버도 시작하지 않는다.

```powershell
python -m pip install -e ".[web]"
$env:CHORD_CHECKPOINT = "models\btc_model.pt"
uvicorn chord_estimator.webapp.app:app --host 127.0.0.1 --port 8000
```

GitHub 업로드와 Render Blueprint 적용 절차, 인스턴스 메모리 선택, 체크포인트 이용 조건은 [docs/DEPLOY_RENDER.md](docs/DEPLOY_RENDER.md)를 따른다. 배포 기본값은 무료 플랜이 아니라 2GB `1c-2g` 플랜이다.

MP3/WAV의 혼합 음원에서 시간별 **코드 심볼**을 추정하는 PyTorch 프로젝트다. 기타 프렛, 보이싱, 운지, 카포 위치를 복원하는 시스템이 아니며 출력에도 이 구분을 기록한다. 체크포인트가 없거나 호환되지 않으면 분석을 중단한다. 무작위 가중치나 고정 코드 진행을 결과로 내보내지 않는다.

## 선택한 구조

기본 모델은 `log-CQT -> 입력 projection -> 비인과적 TransformerEncoder -> frame-wise chord head`다. self-attention이 과거와 미래 문맥을 모두 보는 구조라서 양방향 화성 문맥을 사용하면서 PyTorch 표준 연산으로 학습·배포할 수 있다. 선택적 `cnn_transformer`는 시간축 stride가 1인 작은 CNN 두 층을 앞에 두어 국소 스펙트럼 패턴을 비교한다. 구조가 달라지므로 Transformer-only 체크포인트와 호환되지 않는다.

자체 모델의 기본 텐서는 다음과 같다.

| 단계 | shape | 설명 |
|---|---:|---|
| 디코딩 PCM | `[samples]` | FFmpeg, mono, 22,050 Hz |
| log-CQT | `[T, 216]` | C1, 36 bins/octave, 216 bins, hop 512 |
| 학습/추론 batch | `[B, T_chunk, 216]` | padding은 별도 `[B,T]` mask |
| encoder hidden | `[B, T_chunk, 192]` | 양방향 문맥 |
| chord logits | `[B, T_chunk, 25]` | 12 roots x major/minor + N |
| 선택적 boundary logits | `[B, T_chunk]` | 별도 추가학습 head |

BTC-ISMIR19 호환 프로필은 원본 설정을 고정한다: `[B,108,144] -> [B,108,25]`, 22,050 Hz, hop 2048, C1, 24 bins/octave, 144 bins, 8 layers, hidden 128, 4 heads. 원본처럼 10초 PCM 블록마다 CQT를 계산하고 `log(abs(CQT)+1e-6)` 및 체크포인트의 scalar mean/std를 사용한다. 장곡은 108-frame 입력을 54-frame overlap으로 추론하고 중앙 가중 결합한다. 원본의 입력 길이·아키텍처·라벨 순서는 바꾸지 않는다.

프레임 기준 sample은 `frame_origin + frame_index * hop_length * time_stride`다. librosa CQT의 centered padding을 학습과 추론에서 동일하게 유지한다. 출력의 `frame_step_sec`는 샘플링 간격이지 실제 코드 경계 정확도가 아니다.

## 프로젝트 구조

```text
src/chord_estimator/
  audio_io.py       FFmpeg decode, mono/resample, source hash
  features.py       log-CQT, BTC preprocessing, train-only normalization, cache
  labels.py         internal vocabulary, display notation, transpose, strict mapping
  dataset.py        manifest validation, annotation alignment, post-split chunks
  models/           custom Transformer/CNN and exact BTC adapter
  training.py       AdamW, masked CE, optional boundary loss, early stopping
  inference.py      batched overlapping long-track inference
  decoding.py       optional Viterbi, uncertainty, JSON/CSV/LAB/raw-frame output
  evaluation.py     duration-weighted accuracy and boundary F1
  pipeline.py       reusable analysis engine (web API adapter boundary)
  cli.py            analyze/train/evaluate
configs/             reproducible experiment settings
docs/                weight and dataset provenance
tests/               timing, labels, padding, chunks, decoding, checkpoints
```

## 설치

Python 3.10-3.13과 FFmpeg가 필요하다. FFmpeg의 `ffmpeg` 실행 파일을 PATH에 넣거나 CLI에 `--ffmpeg`를 지정한다.

```powershell
cd C:\tmp\mir_chord_estimator
conda create -n mir-chords python=3.12 -y
conda activate mir-chords
conda install -c conda-forge ffmpeg -y
python -m pip install -e ".[eval,test]"
chord-estimator --help
```

현재 머신의 기본 Anaconda에는 PyTorch만 있고 FFmpeg/librosa/mir_eval은 없으므로 실제 MP3 분석 전에 위 설치가 필요하다.

## 데이터 준비

정답은 `.lab`의 `start_sec end_sec chord` 또는 CSV의 `start_sec,end_sec,chord`다. manifest는 JSON Lines이며 split을 **녹음 단위로 먼저** 정한 결과다.

```json
{"recording_id":"gs_00_comp_01","group_id":"gs_00_piece_01","audio":"D:/data/audio/00_comp_01.wav","annotation":"D:/data/lab/00_comp_01.lab","split":"train","source_type":"guitar"}
{"recording_id":"mix_0001","group_id":"mix_0001","audio":"D:/data/audio/mix_0001.mp3","annotation":"D:/data/lab/mix_0001.lab","split":"validation","source_type":"mixture"}
{"recording_id":"mix_0100","group_id":"mix_0100","audio":"D:/data/audio/mix_0100.mp3","annotation":"D:/data/lab/mix_0100.lab","split":"test","source_type":"mixture"}
```

`group_id`에는 동일 연주의 mic/hex/mix 채널, 다른 인코딩, 원본과 증강본을 모두 같은 값으로 넣는다. loader는 동일 `group_id`, 동일 audio path, 중복 `recording_id`의 split 누출을 거부한다. 청크는 split 검증 이후 생성된다. 실제로 같은 녹음인지 알 수 없는 파일은 도구가 자동 판별할 수 없으므로 데이터 제작자가 `group_id`를 책임지고 부여해야 한다.

25-class 매핑은 major/minor와 N만 지원한다. `C:7`, `C:maj7`, `C:sus4`, `X` 같은 정답은 N이 아니라 `ignore_index=-100`으로 mask되고 평가 보고서에 미지원 비율로 기록된다. N은 실제 무화성 구간이며 UNKNOWN은 추론 점수가 낮아 검토가 필요한 결과다.

피치 이동 증강 시 `transpose_internal()`로 라벨을 같은 semitone만큼 이동해야 한다. time stretch 시 오디오 배율과 모든 start/end 시각을 함께 변환해야 한다. 이 버전은 잘못된 자동 증강을 피하기 위해 학습 CLI에서 증강을 기본 제공하지 않는다.

## 학습

```powershell
chord-estimator train `
  --config configs/custom_25.yaml `
  --manifest D:\data\manifest.jsonl `
  --output-dir runs\baseline `
  --device cuda
```

CNN 비교는 `configs/custom_cnn_25.yaml`을 사용한다. checkpoint에는 state dict, model/feature 설정, 라벨 순서, train split에서만 계산한 mean/std, split manifest SHA-256, epoch, provenance, optimizer state를 저장한다. early stopping 기준은 validation CE다.

검증한 BTC 가중치를 동일 아키텍처와 원본 normalization으로 추가 학습할 수 있다. config에서는 optimizer/epoch/batch 설정을 사용하고 feature/model 설정은 BTC 고정값으로 대체된다.

```powershell
chord-estimator train `
  --config configs/custom_25.yaml `
  --manifest D:\data\manifest.jsonl `
  --init-btc-checkpoint models\btc_model.pt `
  --output-dir runs\btc_finetune
```

## 분석

자체 체크포인트:

```powershell
chord-estimator analyze song.mp3 `
  --checkpoint runs\baseline\best.pt `
  --output outputs\song `
  --device auto `
  --cache-dir cache
```

BTC 호환 체크포인트:

```powershell
chord-estimator analyze song.mp3 `
  --profile btc-ismir19-25 `
  --checkpoint models\btc_model.pt `
  --output outputs\song_btc
```

생성물은 `.json`, `.csv`, `.lab`, `.frames.npz`다. JSON/CSV의 `score`는 **보정되지 않은 softmax score**이고 정답 확률로 해석하면 안 된다. normalized entropy가 `uncertainty`이며 threshold 미만은 UNKNOWN/`needs_review=true`가 된다. `.frames.npz`에는 겹침 결합으로 복원한 전체 raw logits와 softmax score가 있다. Viterbi는 기본 off이며 `--viterbi-stay-bias`를 validation에서 조정한 경우에만 켠다.

## 평가

```powershell
chord-estimator evaluate `
  --manifest D:\data\manifest.jsonl `
  --checkpoint runs\baseline\best.pt `
  --split test `
  --boundary-tolerance-sec 0.5 `
  --output reports\test.json
```

보고서는 시간 가중 25-class 정확도(N 포함/제외), boundary precision/recall/F1, 허용 오차, 미지원 정답 비율, guitar/mixture strata, 곡별 decode/feature/inference 시간과 장치 정보를 포함한다. 확보하지 않은 데이터의 정확도나 실행하지 않은 하드웨어 속도를 채우지 않는다.

## 제한

- 이 모델의 출력은 코드명이며 실제 기타의 운지·프렛·보이싱·카포를 식별하지 않는다.
- 25-class 모델은 7th/sus/extension을 표현하지 못한다. 확장은 정답 vocabulary와 checkpoint label order를 함께 새 버전으로 정의해야 한다.
- boundary head는 구현되어 있으나 `boundary_head: true`로 새로 학습해야 한다. 기존 체크포인트가 자동 호환된다고 가정하지 않는다.
- 합성 오디오는 보조 학습 자료일 뿐 실제 혼합 음악의 held-out test 결과를 대신하지 않는다.
