from __future__ import annotations

import json
from pathlib import Path
import time

from fastapi.testclient import TestClient

from chord_estimator.webapp.app import create_app


class FakeEngine:
    profile = "test-profile"
    checkpoint_sha256 = "a" * 64

    def analyze(self, _input_path, output_prefix, original_filename, progress):
        progress("inference", 70)
        prefix = Path(output_prefix)
        paths = {
            "json": str(prefix.with_suffix(".json")),
            "csv": str(prefix.with_suffix(".csv")),
            "lab": str(prefix.with_suffix(".lab")),
            "frames": str(prefix.with_suffix(".frames.npz")),
        }
        metadata = {
            "audio": {"filename": original_filename, "duration_sec": 1.0},
            "timing": {"inference_sec": 0.01},
        }
        segment = {
            "start_sec": 0.0,
            "end_sec": 1.0,
            "chord": "C:maj",
            "display": "C",
            "candidates": [{"label": "C:maj", "display": "C", "score": 0.8}],
            "score": 0.8,
            "uncertainty": 0.2,
            "needs_review": False,
        }
        Path(paths["json"]).write_text(
            json.dumps({"metadata": metadata, "segments": [segment]}), encoding="utf-8"
        )
        Path(paths["csv"]).write_text("start_sec,end_sec,chord\n0,1,C:maj\n", encoding="utf-8")
        Path(paths["lab"]).write_text("0\t1\tC:maj\n", encoding="utf-8")
        Path(paths["frames"]).write_bytes(b"test-npz")
        return {"paths": paths, "frames": 1, "segments": 1, "chunk_starts": 1}


def test_web_job_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("CHORD_WORK_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    app = create_app(FakeEngine())
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["profile"] == "test-profile"
        assert client.post("/api/jobs", files={"file": ("bad.txt", b"x")}).status_code == 415

        created = client.post("/api/jobs", files={"file": ("sample.mp3", b"fake-audio")})
        assert created.status_code == 202
        job_id = created.json()["id"]
        for _ in range(100):
            payload = client.get(f"/api/jobs/{job_id}").json()
            if payload["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert payload["status"] == "succeeded"
        assert payload["result"]["segments"][0]["display"] == "C"
        assert client.get(payload["downloads"]["json"]).status_code == 200
        assert client.get(payload["audio_url"]).content == b"fake-audio"
