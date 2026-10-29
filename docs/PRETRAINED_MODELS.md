# Pretrained model audit

## BTC-ISMIR19 25-class

- Upstream: `jayg996/BTC-ISMIR19`
- Pinned commit: `2682317be668032e6e4b269ded36adaa2ad57df0`
- File: `test/btc_model.pt` (GitHub shows 11.6 MB)
- Locally verified size: `12,154,754` bytes
- Locally verified SHA-256: `71c2c5db17e8c43b8a9a9da5db36ef2d667158c07a214eba16344c154c00bf54`
- Stable page: <https://github.com/jayg996/BTC-ISMIR19/blob/2682317be668032e6e4b269ded36adaa2ad57df0/test/btc_model.pt>
- Architecture/config source: `run_config.yaml`, `btc_model.py`, `utils/transformer_modules.py`, `utils/mir_eval_modules.py`
- Source code license: MIT, Copyright (c) 2019 Jonggwon Park.
- Checkpoint license: upstream does **not** state a separate checkpoint/model-weight license. Do not assume the source-code MIT grant resolves training-data or model-weight rights. Obtain legal approval before redistribution or commercial use.
- Training data stated upstream: Isophonics, Robbie Williams, UsPop2002 chord labels; their audio is not distributed by the repository and was collected separately from online music services.
- Checkpoint payload expected: `model`, `mean`, `std`.
- Label order: `C:maj,C:min,C#:maj,C#:min,...,B:maj,B:min,N`.
- Required preprocessing: mono 22050 Hz; independent 10 s CQT blocks; default C1; `n_bins=144`; `bins_per_octave=24`; `hop_length=2048`; `log(abs(CQT)+1e-6)`; checkpoint scalar mean/std.
- Input: exactly 108 frames; output: 25 logits per frame.

The project intentionally does not silently download this checkpoint. After reviewing the conditions, download the exact pinned file yourself:

```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/2682317be668032e6e4b269ded36adaa2ad57df0/test/btc_model.pt" `
  -OutFile "models\btc_model.pt"
Get-FileHash -Algorithm SHA256 "models\btc_model.pt"
```

Confirm that the resulting SHA-256 matches the locally verified value above and record it in the experiment log.

The Render Docker build is the only opt-in exception: `Dockerfile` calls
`scripts/fetch_btc_model.py --accept-unverified-weight-terms` with the exact pinned
URL and hash above. Remove that build step if you have not approved the checkpoint
terms for your deployment. The flag documents an explicit decision; it is not a
claim that the model-weight license has been resolved.

The 170-class upstream file also exists, but this implementation does not connect it because its expanded mapping and end-to-end adapter have not been validated here. Passing it to the 25-class profile fails state-shape validation instead of returning a false result.
