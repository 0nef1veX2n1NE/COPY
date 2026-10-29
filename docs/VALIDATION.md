# Validation record

Date: 2026-10-29 (Asia/Seoul)

## Automated tests

- Command: `python -m pytest tests -q`
- Result: `18 passed`
- Covered: internal/display labels, N vs UNKNOWN, unsupported-label masking, transposition, linear and BTC block sample coordinates, final-duration clamp, chunk coverage/overlap-add, padding/model shapes, optional CNN and boundary heads, segment merge/top-k/uncertainty, checkpoint round-trip/missing-checkpoint stop, manifest group leakage.
- Non-failing warning: PyTorch disables its nested-tensor fast path for `norm_first=True`; this does not change outputs.

## BTC checkpoint

- Pinned commit: `2682317be668032e6e4b269ded36adaa2ad57df0`
- File size: 12,154,754 bytes
- SHA-256: `71c2c5db17e8c43b8a9a9da5db36ef2d667158c07a214eba16344c154c00bf54`
- Strict state-dict load: passed
- Forward check: `[1,108,144] -> [1,108,25]`
- Parameters: 3,018,137
- Checkpoint normalization read: mean `-2.2279364109039306`, std `1.719106559064301`

The upstream source is MIT. The checkpoint itself has no separate upstream license statement; local validation is not a claim that redistribution or commercial use is permitted.

## MP3 end-to-end

Input: upstream `test/example.mp3`, SHA-256 `3a2f7aa3a9d1dce25b455b3c85c7661dae39355c1c1ca33b5c42d2820696bff9`.

- Decoded: mono, 22,050 Hz, 257.2014966 s, no trim (`offset=0`)
- Frames: 2,778 x 25; all logits finite
- Postprocessed segments: 207
- Final segment end: 257.2014966 s (matches decoded duration)
- Files written: JSON, CSV, LAB, compressed raw-frame NPZ
- Second-run cache check: `cache_hit=true`; feature load 0.0088 s in that run

Uncached warm-process measurement:

- CPU: AMD64 Family 26 Model 96 Stepping 0, AuthenticAMD
- OS: Windows 11 build 26200
- Python: 3.13.9
- PyTorch: 2.11.0+cu128, inference device CPU
- Model load: 0.0417 s
- FFmpeg decode: 0.2971 s
- log-CQT: 5.3539 s (the first-ever librosa/Numba run was 65.2898 s due to JIT compilation)
- Model inference: 0.5744 s
- Peak process working set: 754,913,280 bytes
- GPU test: not executed

These are measurements of this one supplied example, not accuracy results. No labeled held-out dataset was present, so weighted chord accuracy and boundary F1 were not fabricated or reported.

