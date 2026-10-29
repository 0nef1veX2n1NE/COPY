from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FeatureConfig
from .labels import LABEL_TO_INDEX_25, map_to_majmin
from .timing import frame_times

IGNORE_INDEX = -100


@dataclass(frozen=True)
class Track:
    recording_id: str
    group_id: str
    audio: str
    annotation: str
    split: str
    source_type: str = "unknown"


@dataclass(frozen=True)
class Interval:
    start_sec: float
    end_sec: float
    chord: str


def load_manifest(path: str | Path) -> tuple[list[Track], str]:
    source = Path(path)
    tracks: list[Track] = []
    raw_bytes = source.read_bytes()
    for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row.setdefault("group_id", row.get("recording_id"))
        row.setdefault("source_type", "unknown")
        track = Track(**row)
        if track.split not in {"train", "validation", "test"}:
            raise ValueError(f"line {line_number}: invalid split {track.split!r}")
        tracks.append(track)
    validate_no_leakage(tracks)
    return tracks, hashlib.sha256(raw_bytes).hexdigest()


def validate_no_leakage(tracks: Iterable[Track]) -> None:
    group_splits: dict[str, str] = {}
    audio_splits: dict[str, str] = {}
    recording_ids: set[str] = set()
    for track in tracks:
        if track.recording_id in recording_ids:
            raise ValueError(f"Duplicate recording_id: {track.recording_id}")
        recording_ids.add(track.recording_id)
        resolved = str(Path(track.audio).resolve()).casefold()
        for key, table, name in ((track.group_id, group_splits, "group_id"), (resolved, audio_splits, "audio path")):
            previous = table.setdefault(key, track.split)
            if previous != track.split:
                raise ValueError(f"Split leakage: {name} {key!r} appears in {previous} and {track.split}")


def read_annotations(path: str | Path) -> list[Interval]:
    source = Path(path)
    rows: list[Interval] = []
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append(Interval(float(row["start_sec"]), float(row["end_sec"]), row["chord"].strip()))
    else:
        for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                raise ValueError(f"{source}:{line_number}: expected start end chord")
            rows.append(Interval(float(parts[0]), float(parts[1]), parts[2].strip()))
    previous_end = 0.0
    for interval in rows:
        if interval.start_sec < 0 or interval.end_sec <= interval.start_sec or interval.start_sec < previous_end - 1e-7:
            raise ValueError(f"Invalid or overlapping interval in {source}: {interval}")
        previous_end = interval.end_sec
    return rows


def align_labels(intervals: list[Interval], n_frames: int, config: FeatureConfig, offset_sec: float = 0.0) -> tuple[np.ndarray, dict[str, int]]:
    times = frame_times(n_frames, config, offset_sec)
    labels = np.full(n_frames, IGNORE_INDEX, np.int64)
    counts = {"supported_frames": 0, "unsupported_frames": 0, "unannotated_frames": n_frames}
    for interval in intervals:
        mask = (times >= interval.start_sec) & (times < interval.end_sec)
        if not np.any(mask):
            continue
        counts["unannotated_frames"] -= int(mask.sum())
        mapping = map_to_majmin(interval.chord)
        if mapping.supported and mapping.internal is not None:
            labels[mask] = LABEL_TO_INDEX_25[mapping.internal]
            counts["supported_frames"] += int(mask.sum())
        else:
            counts["unsupported_frames"] += int(mask.sum())
    return labels, counts


@dataclass
class PreparedTrack:
    features: np.ndarray
    labels: np.ndarray
    recording_id: str


class ChunkDataset(Dataset):
    def __init__(self, tracks: list[PreparedTrack], chunk_frames: int, hop_frames: int):
        self.tracks = tracks
        self.chunk_frames = chunk_frames
        self.items: list[tuple[int, int]] = []
        for track_index, track in enumerate(tracks):
            starts = list(range(0, max(len(track.features) - chunk_frames + 1, 1), hop_frames))
            last = max(0, len(track.features) - chunk_frames)
            if not starts or starts[-1] != last:
                starts.append(last)
            self.items.extend((track_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        track_index, start = self.items[index]
        track = self.tracks[track_index]
        length = min(self.chunk_frames, len(track.features) - start)
        features = np.zeros((self.chunk_frames, track.features.shape[1]), np.float32)
        labels = np.full(self.chunk_frames, IGNORE_INDEX, np.int64)
        valid = np.zeros(self.chunk_frames, bool)
        features[:length] = track.features[start:start + length]
        labels[:length] = track.labels[start:start + length]
        valid[:length] = True
        boundary = np.zeros(self.chunk_frames, np.float32)
        if length > 1:
            comparable = (labels[1:length] != IGNORE_INDEX) & (labels[:length - 1] != IGNORE_INDEX)
            boundary[1:length] = ((labels[1:length] != labels[:length - 1]) & comparable).astype(np.float32)
        return {
            "features": torch.from_numpy(features), "labels": torch.from_numpy(labels),
            "valid": torch.from_numpy(valid), "boundary": torch.from_numpy(boundary),
        }

