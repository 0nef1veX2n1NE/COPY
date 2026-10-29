import numpy as np

from chord_estimator.chunking import chunk_starts, overlap_add


def test_chunk_starts_cover_tail_without_duplication_gap():
    assert chunk_starts(20, 8, 4) == [0, 4, 8, 12]
    assert chunk_starts(21, 8, 4) == [0, 4, 8, 12, 13]
    assert chunk_starts(5, 8, 4) == [0]


def test_overlap_add_reconstructs_identical_frame_values():
    source = np.arange(42, dtype=np.float32).reshape(21, 2)
    starts = chunk_starts(len(source), 8, 4)
    chunks = [source[start:start + 8] for start in starts]
    np.testing.assert_allclose(overlap_add(chunks, starts, len(source)), source, rtol=1e-6, atol=1e-6)

