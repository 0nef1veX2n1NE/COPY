from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import threading
import time
import uuid
from typing import Any

from .engine import AnalyzerEngine


@dataclass
class JobRecord:
    id: str
    original_filename: str
    directory: Path
    input_path: Path
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "uploading"
    stage: str = "upload"
    progress: int = 0
    error: str | None = None
    outputs: dict[str, str] | None = None
    summary: dict[str, Any] | None = None


class JobStore:
    def __init__(self, engine: AnalyzerEngine, root: str | Path, ttl_seconds: int = 3600, max_pending: int = 2):
        self.engine = engine
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.max_pending = max_pending
        self.records: dict[str, JobRecord] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chord-analyzer")

    def reserve(self, filename: str, suffix: str) -> JobRecord:
        self.cleanup_expired()
        with self.lock:
            pending = sum(record.status in {"uploading", "queued", "running"} for record in self.records.values())
            if pending >= self.max_pending:
                raise OverflowError("분석 대기열이 가득 찼습니다. 잠시 후 다시 시도해 주세요.")
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir(parents=True, exist_ok=False)
            record = JobRecord(job_id, filename, directory, directory / f"input{suffix}")
            self.records[job_id] = record
            return record

    def discard(self, job_id: str) -> None:
        with self.lock:
            record = self.records.pop(job_id, None)
        if record and record.directory.is_relative_to(self.root):
            shutil.rmtree(record.directory, ignore_errors=True)

    def enqueue(self, job_id: str) -> None:
        self._update(job_id, status="queued", stage="queued", progress=5)
        self.executor.submit(self._run, job_id)

    def _update(self, job_id: str, **values) -> None:
        with self.lock:
            record = self.records[job_id]
            for key, value in values.items():
                setattr(record, key, value)
            record.updated_at = time.time()

    def _run(self, job_id: str) -> None:
        self._update(job_id, status="running", stage="starting", progress=8)
        with self.lock:
            record = self.records[job_id]
        try:
            result = self.engine.analyze(
                record.input_path,
                record.directory / "result",
                record.original_filename,
                lambda stage, progress: self._update(job_id, stage=stage, progress=progress),
            )
            document = json.loads(Path(result["paths"]["json"]).read_text(encoding="utf-8"))
            summary = {
                "metadata": document["metadata"],
                "segments": document["segments"],
                "segment_count": len(document["segments"]),
                "frame_count": result["frames"],
            }
            self._update(job_id, status="succeeded", stage="complete", progress=100, outputs=result["paths"], summary=summary)
        except Exception as error:
            self._update(job_id, status="failed", stage="failed", error=str(error), progress=100)

    def public(self, job_id: str) -> dict[str, Any]:
        self.cleanup_expired()
        with self.lock:
            record = self.records.get(job_id)
            if record is None:
                raise KeyError(job_id)
            result = {
                "id": record.id,
                "filename": record.original_filename,
                "status": record.status,
                "stage": record.stage,
                "progress": record.progress,
                "error": record.error,
                "created_at": record.created_at,
            }
            if record.status == "succeeded":
                result["result"] = record.summary
                result["downloads"] = {kind: f"/api/jobs/{job_id}/download/{kind}" for kind in (record.outputs or {})}
                result["audio_url"] = f"/api/jobs/{job_id}/audio"
            return result

    def path_for(self, job_id: str, kind: str) -> Path:
        with self.lock:
            record = self.records.get(job_id)
            if not record or record.status != "succeeded" or not record.outputs or kind not in record.outputs:
                raise KeyError(job_id)
            path = Path(record.outputs[kind]).resolve()
            if not path.is_relative_to(record.directory.resolve()):
                raise RuntimeError("Unsafe output path")
            return path

    def audio_for(self, job_id: str) -> tuple[Path, str]:
        with self.lock:
            record = self.records.get(job_id)
            if not record or record.status != "succeeded":
                raise KeyError(job_id)
            return record.input_path, record.original_filename

    def cleanup_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        with self.lock:
            expired = [record.id for record in self.records.values() if record.updated_at < cutoff and record.status not in {"uploading", "queued", "running"}]
        for job_id in expired:
            self.discard(job_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

