from __future__ import annotations

from dataclasses import asdict
import os
import platform
from pathlib import Path
import time
from typing import Any

import torch

from .audio_io import decode_audio, sha256_file
from .checkpoint import load_inference_checkpoint
from .decoding import decode_segments, write_outputs
from .features import compute_btc_legacy_log_cqt, compute_log_cqt, feature_cache_key, load_or_compute_cache, normalize_features
from .inference import infer_feature_sequence
from .labels import LABELS_25


def _process_memory() -> dict[str, int] | None:
    try:
        import psutil
        info = psutil.Process().memory_info()
        result = {"rss_bytes": int(info.rss)}
        peak = getattr(info, "peak_wset", None)
        if peak is not None:
            result["peak_working_set_bytes"] = int(peak)
        return result
    except (ImportError, OSError):
        return None


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def analyze_audio(
    audio_path: str, checkpoint_path: str, device: torch.device,
    chunk_frames: int | None = None, chunk_hop_frames: int | None = None, batch_size: int = 8,
    review_threshold: float = 0.45, top_k: int = 3,
    viterbi_stay_bias: float | None = None, cache_dir: str | None = None,
    ffmpeg: str | None = None, profile: str = "custom",
) -> dict[str, Any]:
    load_start = time.perf_counter()
    model, checkpoint = load_inference_checkpoint(checkpoint_path, device, profile)
    model_load_sec = time.perf_counter() - load_start
    feature_config = checkpoint["feature_config_object"]
    if profile == "btc-ismir19-25":
        chunk_frames = 108 if chunk_frames is None else chunk_frames
        chunk_hop_frames = 54 if chunk_hop_frames is None else chunk_hop_frames
        if chunk_frames != 108:
            raise ValueError("BTC-ISMIR19 input length is fixed at exactly 108 frames")
        feature_function = compute_btc_legacy_log_cqt
    else:
        chunk_frames = 512 if chunk_frames is None else chunk_frames
        chunk_hop_frames = 256 if chunk_hop_frames is None else chunk_hop_frames
        feature_function = compute_log_cqt
    decode_start = time.perf_counter()
    audio = decode_audio(audio_path, feature_config.sample_rate, ffmpeg)
    decode_sec = time.perf_counter() - decode_start
    checkpoint_sha = sha256_file(checkpoint_path)
    analysis_settings = {"chunk_frames": chunk_frames, "chunk_hop_frames": chunk_hop_frames}
    cache_key = feature_cache_key(audio.source_sha256, feature_config, checkpoint_sha, analysis_settings)
    feature_start = time.perf_counter()
    features, cache_hit = load_or_compute_cache(cache_dir, cache_key, lambda: feature_function(audio.samples, feature_config))
    feature_sec = time.perf_counter() - feature_start
    stats = checkpoint["normalization"]
    normalized = normalize_features(features, stats["mean"], stats["std"])
    logits, inference_extra, timings = infer_feature_sequence(
        model, normalized, device, chunk_frames, chunk_hop_frames, batch_size,
    )
    segments, probabilities = decode_segments(
        logits, LABELS_25, feature_config, audio.duration_sec, audio.offset_sec,
        top_k, review_threshold, viterbi_stay_bias,
    )
    metadata = {
        "audio": {"path": audio.source_path, "sha256": audio.source_sha256, "decoded_sample_rate": audio.sample_rate, "duration_sec": audio.duration_sec, "trim_offset_sec": audio.offset_sec},
        "checkpoint": {"path": str(Path(checkpoint_path).resolve()), "sha256": checkpoint_sha, "profile": profile, "format_version": checkpoint["format_version"], "epoch": checkpoint.get("epoch"), "split_version": checkpoint.get("split_version"), "provenance": checkpoint.get("provenance")},
        "feature_config": asdict(feature_config),
        "normalization": {"source": "checkpoint/training-split", "mean": stats["mean"], "std": stats["std"]},
        "analysis": {**analysis_settings, "top_k": top_k, "review_threshold": review_threshold, "viterbi_stay_bias": viterbi_stay_bias, "cache_key": cache_key, "cache_hit": cache_hit, "nominal_frame_step_sec": feature_config.hop_length * feature_config.time_stride / feature_config.sample_rate, "block_samples": feature_config.block_samples, "frames_per_block": feature_config.frames_per_block, "frame_step_is_not_boundary_accuracy": True},
        "timing": {"model_load_sec": model_load_sec, "decode_sec": decode_sec, "feature_sec": feature_sec, "inference_sec": timings.inference_sec, "peak_cuda_bytes": timings.peak_cuda_bytes, "process_memory": _process_memory()},
        "runtime": {"device": str(device), "python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"), "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
    }
    return {"segments": segments, "logits": logits, "probabilities": probabilities, "metadata": metadata, "extra": inference_extra, "feature_config": feature_config}


def analyze_to_files(audio_path: str, output_prefix: str, **kwargs) -> dict[str, Any]:
    result = analyze_audio(audio_path=audio_path, **kwargs)
    paths = write_outputs(output_prefix, result["segments"], result["logits"], result["probabilities"], result["metadata"], LABELS_25)
    return {"outputs": paths, "metadata": result["metadata"], "segments": len(result["segments"])}
