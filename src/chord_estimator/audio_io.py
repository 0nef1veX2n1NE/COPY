from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import subprocess

import numpy as np


class AudioDecodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray
    sample_rate: int
    source_path: str
    source_sha256: str
    offset_sec: float = 0.0

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / float(self.sample_rate)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_ffmpeg(executable: str | None = None) -> str:
    resolved = executable or shutil.which("ffmpeg")
    if not resolved:
        raise AudioDecodeError(
            "FFmpeg was not found. Install it and put ffmpeg on PATH, or pass --ffmpeg PATH."
        )
    return resolved


def decode_audio(path: str | Path, sample_rate: int, ffmpeg: str | None = None) -> DecodedAudio:
    """Decode once to mono float32 PCM. No silence trimming is performed."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    command = [
        require_ffmpeg(ffmpeg), "-nostdin", "-v", "error", "-i", str(source),
        "-map_metadata", "-1", "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"FFmpeg decode failed ({process.returncode}): {detail}")
    samples = np.frombuffer(process.stdout, dtype="<f4").copy()
    if samples.size == 0:
        raise AudioDecodeError("Decoded audio is empty")
    return DecodedAudio(samples, sample_rate, str(source), sha256_file(source), 0.0)

