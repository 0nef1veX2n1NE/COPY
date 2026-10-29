from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable

import torch

from ..audio_io import decode_audio, sha256_file
from ..checkpoint import load_inference_checkpoint
from ..decoding import decode_segments, write_outputs
from ..features import (
    compute_btc_legacy_log_cqt,
    compute_log_cqt,
    feature_cache_key,
    load_or_compute_cache,
    normalize_features,
)
from ..inference import infer_feature_sequence
from ..labels import LABELS_25
from ..pipeline import _process_memory, choose_device

ProgressCallback = Callable[[str, int], None]


class AnalyzerEngine:
    """Long-lived, single-model analyzer used by the web worker."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        profile: str = "btc-ismir19-25",
        device_name: str = "cpu",
        cache_dir: str | Path | None = None,
        ffmpeg: str | None = None,
        batch_size: int = 8,
    ):
        thread_count = int(os.environ.get("TORCH_NUM_THREADS", "0"))
        if thread_count > 0:
            torch.set_num_threads(thread_count)
        self.device = choose_device(device_name)
        self.profile = profile
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.model, self.checkpoint = load_inference_checkpoint(self.checkpoint_path, self.device, profile)
        self.feature_config = self.checkpoint["feature_config_object"]
        self.normalization = self.checkpoint["normalization"]
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.ffmpeg = ffmpeg
        self.batch_size = batch_size
        if profile == "btc-ismir19-25":
            self.chunk_frames = 108
            self.chunk_hop_frames = 54
            self.feature_function = compute_btc_legacy_log_cqt
        else:
            self.chunk_frames = 512
            self.chunk_hop_frames = 256
            self.feature_function = compute_log_cqt

    def analyze(
        self,
        input_path: str | Path,
        output_prefix: str | Path,
        original_filename: str,
        progress: ProgressCallback,
        review_threshold: float = 0.45,
        top_k: int = 3,
    ) -> dict[str, Any]:
        progress("decode", 12)
        decode_started = time.perf_counter()
        audio = decode_audio(input_path, self.feature_config.sample_rate, self.ffmpeg)
        decode_sec = time.perf_counter() - decode_started

        settings = {"chunk_frames": self.chunk_frames, "chunk_hop_frames": self.chunk_hop_frames}
        key = feature_cache_key(audio.source_sha256, self.feature_config, self.checkpoint_sha256, settings)
        progress("log-cqt", 30)
        feature_started = time.perf_counter()
        features, cache_hit = load_or_compute_cache(
            self.cache_dir,
            key,
            lambda: self.feature_function(audio.samples, self.feature_config),
        )
        feature_sec = time.perf_counter() - feature_started
        normalized = normalize_features(features, self.normalization["mean"], self.normalization["std"])

        progress("inference", 68)
        logits, extra, inference_timing = infer_feature_sequence(
            self.model,
            normalized,
            self.device,
            self.chunk_frames,
            self.chunk_hop_frames,
            self.batch_size,
        )
        progress("decoding", 88)
        segments, probabilities = decode_segments(
            logits,
            LABELS_25,
            self.feature_config,
            audio.duration_sec,
            audio.offset_sec,
            top_k=top_k,
            review_threshold=review_threshold,
        )
        metadata = {
            "audio": {
                "filename": original_filename,
                "sha256": audio.source_sha256,
                "decoded_sample_rate": audio.sample_rate,
                "duration_sec": audio.duration_sec,
                "trim_offset_sec": audio.offset_sec,
            },
            "checkpoint": {
                "profile": self.profile,
                "sha256": self.checkpoint_sha256,
                "format_version": self.checkpoint["format_version"],
                "epoch": self.checkpoint.get("epoch"),
                "split_version": self.checkpoint.get("split_version"),
                "provenance": self.checkpoint.get("provenance"),
            },
            "feature_config": asdict(self.feature_config),
            "normalization": {
                "source": "checkpoint/training-split",
                "mean": self.normalization["mean"],
                "std": self.normalization["std"],
            },
            "analysis": {
                **settings,
                "top_k": top_k,
                "review_threshold": review_threshold,
                "cache_hit": cache_hit,
                "nominal_frame_step_sec": self.feature_config.hop_length / self.feature_config.sample_rate,
                "frame_step_is_not_boundary_accuracy": True,
            },
            "timing": {
                "decode_sec": decode_sec,
                "feature_sec": feature_sec,
                "inference_sec": inference_timing.inference_sec,
                "peak_cuda_bytes": inference_timing.peak_cuda_bytes,
                "process_memory": _process_memory(),
            },
            "runtime": {
                "device": str(self.device),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
                "cuda_device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            },
        }
        progress("writing", 96)
        paths = write_outputs(output_prefix, segments, logits, probabilities, metadata, LABELS_25)
        return {
            "paths": paths,
            "metadata": metadata,
            "segments": len(segments),
            "frames": int(logits.shape[0]),
            "chunk_starts": len(extra["chunk_starts"]),
        }
