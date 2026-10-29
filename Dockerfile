FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHORD_CHECKPOINT=/app/models/btc_model.pt \
    CHORD_PROFILE=btc-ismir19-25 \
    CHORD_DEVICE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts/fetch_btc_model.py ./scripts/fetch_btc_model.py

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install ".[web]"

ARG BTC_MODEL_URL=https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/2682317be668032e6e4b269ded36adaa2ad57df0/test/btc_model.pt
ARG BTC_MODEL_SHA256=71c2c5db17e8c43b8a9a9da5db36ef2d667158c07a214eba16344c154c00bf54
RUN python scripts/fetch_btc_model.py models/btc_model.pt \
      --url "${BTC_MODEL_URL}" \
      --sha256 "${BTC_MODEL_SHA256}" \
      --accept-unverified-weight-terms

COPY configs ./configs
COPY docs ./docs

RUN useradd --create-home --uid 10001 chordapp \
    && mkdir -p /tmp/chord-cache /tmp/chord-jobs \
    && chown -R chordapp:chordapp /app /tmp/chord-cache /tmp/chord-jobs

USER chordapp
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','10000')+'/healthz', timeout=4)" || exit 1

CMD ["sh", "-c", "exec uvicorn chord_estimator.webapp.app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
