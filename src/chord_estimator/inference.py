from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch

from .chunking import chunk_starts, overlap_add


@dataclass(frozen=True)
class InferenceTimings:
    inference_sec: float
    device: str
    peak_cuda_bytes: int | None


def infer_feature_sequence(
    model: torch.nn.Module, features: np.ndarray, device: torch.device,
    chunk_frames: int, hop_frames: int, batch_size: int = 8,
) -> tuple[np.ndarray, dict[str, Any], InferenceTimings]:
    """Infer overlapping chunks and reconstruct every source frame once."""
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must be non-empty [time, frequency]")
    starts = chunk_starts(len(features), chunk_frames, hop_frames)
    chord_chunks: list[np.ndarray] = []
    boundary_chunks: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(starts), batch_size):
            selected = starts[batch_start:batch_start + batch_size]
            data = np.zeros((len(selected), chunk_frames, features.shape[1]), np.float32)
            valid = np.zeros((len(selected), chunk_frames), bool)
            lengths: list[int] = []
            for row, start in enumerate(selected):
                length = min(chunk_frames, len(features) - start)
                data[row, :length] = features[start:start + length]
                valid[row, :length] = True
                lengths.append(length)
            output = model(
                torch.from_numpy(data).to(device),
                torch.from_numpy(valid).to(device),
            )
            logits = output["chord_logits"].float().cpu().numpy()
            boundaries = output.get("boundary_logits")
            boundary_np = boundaries.float().cpu().numpy() if boundaries is not None else None
            for row, length in enumerate(lengths):
                chord_chunks.append(logits[row, :length])
                if boundary_np is not None:
                    boundary_chunks.append(boundary_np[row, :length, None])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - begin
    reconstructed = overlap_add(chord_chunks, starts, len(features))
    extra: dict[str, Any] = {"chunk_starts": starts}
    if boundary_chunks:
        extra["boundary_logits"] = overlap_add(boundary_chunks, starts, len(features))[:, 0]
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return reconstructed, extra, InferenceTimings(elapsed, str(device), peak)

