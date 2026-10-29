from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import FeatureConfig


def _fmin_hz(value: str | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    import librosa
    return float(librosa.note_to_hz(value))


def compute_log_cqt(samples: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Return [time, frequency] log-magnitude CQT."""
    import librosa
    cqt = librosa.cqt(
        y=np.asarray(samples, dtype=np.float32), sr=config.sample_rate,
        hop_length=config.hop_length, fmin=_fmin_hz(config.fmin),
        n_bins=config.n_bins, bins_per_octave=config.bins_per_octave,
        pad_mode=config.pad_mode,
    )
    return np.log(np.abs(cqt).astype(np.float32) + config.log_epsilon).T


def compute_btc_legacy_log_cqt(samples: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Reproduce upstream BTC's independent 10-second CQT blocks."""
    if config.sample_rate != 22050 or config.hop_length != 2048 or config.n_bins != 144 or config.bins_per_octave != 24:
        raise ValueError("BTC legacy preprocessing requires sr=22050, hop=2048, n_bins=144, bins/octave=24")
    block = 10 * config.sample_rate
    pieces = [compute_log_cqt(samples[start:start + block], config) for start in range(0, len(samples), block)]
    return np.concatenate(pieces, axis=0) if pieces else np.empty((0, config.n_bins), np.float32)


def normalize_features(features: np.ndarray, mean: float | np.ndarray, std: float | np.ndarray) -> np.ndarray:
    std_array = np.asarray(std)
    if np.any(std_array <= 0):
        raise ValueError("Normalization std must be positive")
    return ((features - mean) / std_array).astype(np.float32)


def training_statistics(feature_arrays: list[np.ndarray]) -> tuple[float, float]:
    """Stable scalar statistics over training features only."""
    count = 0
    mean = 0.0
    m2 = 0.0
    for array in feature_arrays:
        flat = np.asarray(array, dtype=np.float64).ravel()
        if flat.size == 0:
            continue
        batch_count = flat.size
        batch_mean = float(flat.mean())
        batch_m2 = float(((flat - batch_mean) ** 2).sum())
        delta = batch_mean - mean
        total = count + batch_count
        mean += delta * batch_count / total
        m2 += batch_m2 + delta * delta * count * batch_count / total
        count = total
    if count < 2:
        raise ValueError("At least two training feature values are required")
    return mean, max((m2 / count) ** 0.5, 1e-8)


def feature_cache_key(audio_sha256: str, config: FeatureConfig, model_version: str, analysis: dict[str, Any]) -> str:
    payload = {
        "audio_sha256": audio_sha256, "feature": asdict(config),
        "model_version": model_version, "analysis": analysis,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_or_compute_cache(
    cache_dir: str | Path | None, key: str, compute,
) -> tuple[np.ndarray, bool]:
    if cache_dir is None:
        return compute(), False
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            return archive["features"], True
    value = compute()
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, features=value)
    temporary.replace(path)
    return value, False

