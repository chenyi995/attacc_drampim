# Superseded v1 timing model

`timing_models_rag.json` is the version-1 model: a 113-bucket lookup table
keyed on twelve fields including `channel_base`, with a piecewise-linear
anchor table per bucket.  It is kept only as the reference for the
measurements in `../RESULTS.md` §4.

It must not be used to price a run.  Loading it now fails deliberately:
`Ramulator` and `run_a1_dag_free_inputs.py` both require `"version": 2`.
Measured behaviour of this file, on the 944 cache entries appended after it
was calibrated:

| prediction path | n | MAPE | p95 | max |
|---|---:|---:|---:|---:|
| memorised anchor | 478 | 0.00% | 0.00% | 0.00% |
| interpolation | 192 | 1.83% | 8.68% | 15.42% |
| extrapolation | 274 | 14.08% | 52.15% | 196.80% |
| bucket absent (silent fallback) | — | ~154% | ~1100% | ~1950% |
