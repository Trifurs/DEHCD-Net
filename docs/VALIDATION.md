# Validation

```bash
python scripts/check_source.py
python -m unittest discover -s tests -v
```

Source checks inspect syntax, private paths, credential-shaped text and Git publication candidates. Dataset files, checkpoints, run records and result tables must remain outside the tracked source tree. Already-tracked local files are rejected even if an ignore rule would hide a new file at that path.

Tests cover model interfaces and gradients, configuration inheritance and references, seed scheduling, metrics, checkpoint recovery, numerical protection, data preparation and output completeness. Temporary fixtures stay in temporary directories. CUDA-specific tests skip when the device or scan extension is unavailable; they do not establish GPU equivalence when skipped.

Use launcher `--dry-run` to inspect a plan, `--audit-only` for read-only checks and `--preflight-only` for data and continuation validation. These modes do not train models. Numerical guards and short execution checks do not establish convergence.

`tools/validate_results.py` validates outputs against saved configuration, checkpoint identity, full split coverage and raw confusion counts. `tools/compare_results.py` obtains the expected repetition set from the saved plan. Missing outputs remain incomplete; duplicate IDs and incompatible source/data/configuration evidence are rejected.
