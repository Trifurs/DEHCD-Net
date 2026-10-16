# Selective-scan CUDA extension

This directory builds the `selective_scan_cuda_oflex` extension used by ChangeMamba. The retained C++/CUDA sources and their license are included in this directory.

Build in the same Python/PyTorch environment used for training, with a matching CUDA toolkit and C++ compiler:

```bash
python -m pip install -r requirements-baselines.txt
python -m pip install --no-build-isolation ./compare/kernels/selective_scan
```

Run these commands from the project root. Set `CUDA_HOME` to the toolkit root when `nvcc` is outside the default location. GPU architecture selection uses PyTorch detection or `TORCH_CUDA_ARCH_LIST`.

See [sources, adaptations and validation](../../UPSTREAM.md) and the retained [license](LICENSE). Forward and backward checks are in `tests/test_damage_baselines.py`; they require an available CUDA device and the compiled extension.
