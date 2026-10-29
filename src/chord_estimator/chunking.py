from __future__ import annotations

import numpy as np


def chunk_starts(n_frames: int, chunk_frames: int, hop_frames: int) -> list[int]:
    if n_frames <= 0 or chunk_frames <= 0 or hop_frames <= 0 or hop_frames > chunk_frames:
        raise ValueError("Invalid frame/chunk/hop count")
    if n_frames <= chunk_frames:
        return [0]
    starts = list(range(0, n_frames - chunk_frames + 1, hop_frames))
    last = n_frames - chunk_frames
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_window(length: int) -> np.ndarray:
    if length == 1:
        return np.ones(1, np.float32)
    window = np.hanning(length + 2)[1:-1].astype(np.float32)
    return np.maximum(window, 1e-4)


def overlap_add(chunks: list[np.ndarray], starts: list[int], n_frames: int) -> np.ndarray:
    if not chunks or len(chunks) != len(starts):
        raise ValueError("chunks and starts must be non-empty and aligned")
    classes = chunks[0].shape[-1]
    total = np.zeros((n_frames, classes), np.float64)
    weight = np.zeros(n_frames, np.float64)
    for chunk, start in zip(chunks, starts):
        valid = min(len(chunk), n_frames - start)
        if valid <= 0:
            continue
        window = blend_window(valid)
        total[start:start + valid] += chunk[:valid] * window[:, None]
        weight[start:start + valid] += window
    if np.any(weight == 0):
        raise RuntimeError("Chunk reconstruction left uncovered frames")
    return (total / weight[:, None]).astype(np.float32)

