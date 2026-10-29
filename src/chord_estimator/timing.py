from __future__ import annotations

import numpy as np
from .config import FeatureConfig


def frame_sample_indices(n_frames: int, config: FeatureConfig) -> np.ndarray:
    indices = np.arange(n_frames, dtype=np.int64)
    if config.block_samples is not None or config.frames_per_block is not None:
        if not config.block_samples or not config.frames_per_block:
            raise ValueError("block_samples and frames_per_block must be set together")
        block = indices // config.frames_per_block
        within = indices % config.frames_per_block
        return config.frame_origin_samples + block * config.block_samples + within * config.hop_length * config.time_stride
    return config.frame_origin_samples + indices * config.hop_length * config.time_stride


def frame_times(n_frames: int, config: FeatureConfig, offset_sec: float = 0.0) -> np.ndarray:
    return frame_sample_indices(n_frames, config) / float(config.sample_rate) + float(offset_sec)


def frame_edges(n_frames: int, config: FeatureConfig, duration_sec: float, offset_sec: float = 0.0) -> np.ndarray:
    edges = offset_sec + frame_sample_indices(n_frames + 1, config) / float(config.sample_rate)
    return np.minimum(edges, offset_sec + duration_sec)
