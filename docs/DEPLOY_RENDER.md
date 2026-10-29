# GitHub → Render 배포

이 서비스는 FastAPI 한 프로세스에서 BTC 모델을 한 번 로딩하고, 한 번에 분석 작업 하나만 실행한다. 업로드와 결과는 `/tmp`에 1시간 동안만 보관된다. Render 재시작 또는 재배포 시 사라지는 임시 데이터이며 사용자 원본의 장기 저장소가 아니다.

## 배포 전에 확인할 사항

1. `docs/PRETRAINED_MODELS.md`의 출처, SHA-256, 전처리와 이용 조건을 읽는다.
2. BTC 저장소는 소스코드 MIT 라이선스를 명시하지만 체크포인트에 대한 별도 라이선스를 명시하지 않는다. 공개·상업 서비스에 쓰기 전에 이용 권한을 확인한다.
3. 실제 Windows CPU 검증에서 프로세스 peak RSS가 약 755MB였으므로 `render.yaml`은 2GB `1c-2g` 인스턴스를 사용한다. 무료 512MB 인스턴스로 낮추는 것은 권장하지 않는다.
4. 웹 서비스에는 로그인·과금·악성 파일 탐지 기능이 없다. 공개 URL을 널리 배포하기 전 프록시 인증과 사용량 제한을 별도로 추가한다.

## 1. GitHub에 올리기

Git과 GitHub CLI를 설치하고 인증한다.

```powershell
winget install --id Git.Git -e
winget install --id GitHub.cli -e
gh auth login
```

새 PowerShell 창에서 다음을 실행한다. 기본값은 비공개 저장소다.

```powershell
cd C:\tmp\mir_chord_estimator
.\scripts\publish_github.ps1 -Repository mir-chord-estimator
```

공개 저장소가 필요하면 `-Visibility public`을 명시한다. `models/btc_model.pt`, 업로드 음원, 결과와 캐시는 `.gitignore`로 제외된다.

## 2. Render Blueprint 생성

1. Render Dashboard에서 **New → Blueprint**를 선택한다.
2. GitHub 연결을 승인하고 위 저장소를 고른다.
3. Render가 루트의 `render.yaml`을 읽으면 서비스 이름, Singapore 리전, Docker 런타임과 `1c-2g` 요금제를 확인한다.
4. 비용을 확인한 뒤 Blueprint를 적용한다.
5. 빌드 로그에서 `Downloaded and verified ... 71c2...bf54`를 확인한다.
6. 배포 후 `https://<service>.onrender.com/healthz`가 `status: ok`를 반환하는지 확인한다.

`autoDeployTrigger: checksPass`이므로 이후 `main` push는 GitHub Actions 테스트가 성공한 뒤 배포된다. Docker 이미지는 고정 커밋 URL의 체크포인트만 받고 SHA-256 불일치 시 빌드를 실패시킨다.

## 운영 설정

중요 환경 변수는 `render.yaml`에 있다.

| 변수 | 기본값 | 의미 |
|---|---:|---|
| `MAX_UPLOAD_MB` | 30 | MP3/WAV 업로드 상한 |
| `MAX_PENDING_JOBS` | 2 | 실행+대기 작업 상한 |
| `JOB_TTL_SECONDS` | 3600 | 완료 결과 보존 시간 |
| `CHORD_BATCH_SIZE` | 8 | CPU 추론 배치 |
| `TORCH_NUM_THREADS` | 2 | PyTorch CPU 스레드 |

수평 확장은 현재 지원하지 않는다. 작업 상태가 프로세스 메모리에 있으므로 `numInstances`는 1로 유지한다. 다중 인스턴스가 필요하면 Redis 기반 작업 큐와 외부 객체 저장소를 먼저 연결한다.

## 로컬 Docker 확인

```powershell
docker build -t chord-trace .
docker run --rm -p 10000:10000 -e PORT=10000 chord-trace
```

그 뒤 <http://localhost:10000>을 연다. Docker 빌드 시 체크포인트 조건을 수락하는 명시적 플래그가 Dockerfile에 있으며, 이는 법률적 승인을 대신하지 않는다.
