from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import AnalyzerEngine
from .jobs import JobStore

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
ALLOWED_SUFFIXES = {".mp3", ".wav"}


def create_app(engine: AnalyzerEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        actual_engine = engine or AnalyzerEngine(
            checkpoint_path=os.environ.get("CHORD_CHECKPOINT", "models/btc_model.pt"),
            profile=os.environ.get("CHORD_PROFILE", "btc-ismir19-25"),
            device_name=os.environ.get("CHORD_DEVICE", "cpu"),
            cache_dir=os.environ.get("CHORD_CACHE_DIR", "/tmp/chord-cache"),
            ffmpeg=os.environ.get("FFMPEG_PATH"),
            batch_size=int(os.environ.get("CHORD_BATCH_SIZE", "8")),
        )
        app.state.engine = actual_engine
        app.state.jobs = JobStore(
            actual_engine,
            os.environ.get("CHORD_WORK_DIR", "/tmp/chord-jobs"),
            ttl_seconds=int(os.environ.get("JOB_TTL_SECONDS", "3600")),
            max_pending=int(os.environ.get("MAX_PENDING_JOBS", "2")),
        )
        yield
        app.state.jobs.shutdown()

    app = FastAPI(
        title="Chord Trace",
        description="Checkpoint-gated BTC chord estimation web service",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    async def health(request: Request):
        engine_instance = request.app.state.engine
        return {
            "status": "ok",
            "profile": engine_instance.profile,
            "checkpoint_sha256": engine_instance.checkpoint_sha256,
        }

    @app.get("/api/info")
    async def info(request: Request):
        engine_instance = request.app.state.engine
        return {
            "profile": engine_instance.profile,
            "labels": 25,
            "max_upload_mb": int(os.environ.get("MAX_UPLOAD_MB", "30")),
            "job_ttl_seconds": int(os.environ.get("JOB_TTL_SECONDS", "3600")),
            "score_semantics": "uncalibrated_softmax_score",
            "scope": "chord-symbol recognition; not fret, voicing, fingering, or capo recovery",
        }

    @app.post("/api/jobs", status_code=202)
    async def create_job(request: Request, file: UploadFile = File(...)):
        filename = Path(file.filename or "upload.mp3").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(415, "MP3 또는 WAV 파일만 업로드할 수 있습니다.")
        store: JobStore = request.app.state.jobs
        try:
            record = store.reserve(filename, suffix)
        except OverflowError as error:
            raise HTTPException(429, str(error)) from error
        limit = int(os.environ.get("MAX_UPLOAD_MB", "30")) * 1024 * 1024
        written = 0
        try:
            with record.input_path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(413, f"파일 크기는 {limit // 1024 // 1024}MB 이하여야 합니다.")
                    output.write(chunk)
            if written == 0:
                raise HTTPException(400, "빈 파일은 분석할 수 없습니다.")
            store.enqueue(record.id)
            return {"id": record.id, "status": "queued", "status_url": f"/api/jobs/{record.id}"}
        except Exception:
            store.discard(record.id)
            raise
        finally:
            await file.close()

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str, request: Request):
        try:
            return request.app.state.jobs.public(job_id)
        except KeyError as error:
            raise HTTPException(404, "작업을 찾을 수 없거나 만료되었습니다.") from error

    @app.get("/api/jobs/{job_id}/download/{kind}")
    async def download(job_id: str, kind: str, request: Request):
        if kind not in {"json", "csv", "lab", "frames"}:
            raise HTTPException(404, "지원하지 않는 결과 형식입니다.")
        try:
            path = request.app.state.jobs.path_for(job_id, kind)
        except KeyError as error:
            raise HTTPException(404, "결과가 없거나 만료되었습니다.") from error
        extension = ".frames.npz" if kind == "frames" else f".{kind}"
        return FileResponse(path, filename=f"chord-trace-{job_id[:8]}{extension}", media_type="application/octet-stream")

    @app.get("/api/jobs/{job_id}/audio")
    async def audio(job_id: str, request: Request):
        try:
            path, filename = request.app.state.jobs.audio_for(job_id)
        except KeyError as error:
            raise HTTPException(404, "오디오가 없거나 만료되었습니다.") from error
        media_type = "audio/mpeg" if path.suffix.lower() == ".mp3" else "audio/wav"
        return FileResponse(path, filename=filename, media_type=media_type)

    @app.exception_handler(500)
    async def internal_error(_request: Request, _error: Exception):
        return JSONResponse(status_code=500, content={"detail": "서버 내부 오류가 발생했습니다."})

    return app


app = create_app()

