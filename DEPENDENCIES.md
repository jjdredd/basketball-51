# Dependency Installation Guide

## Required Packages

| Package | Version | Purpose |
| --------- | --------- | --------- |
| torch | 2.13.0+ | Deep learning framework |
| torchvision | 0.28.0+ | Image/video transforms |
| mmcv-lite | >=2.0.0rc4, <2.2.0 | OpenMMLab computer vision tools |
| mmengine | >=0.3.0 | OpenMMLab training engine |
| mmaction2 | 1.2.0 | Action recognition toolkit |
| decord | 0.6.0 | Video decoding |

## Installation Steps

### 1. Create a Python virtual environment

```bash
cd /home/agentslop/slop/mmaction/basketball51-project
python3 -m venv venv
```

### 2. Activate the virtual environment

```bash
source venv/bin/activate
```

### 3. Install PyTorch and TorchVision (CPU version)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

*For GPU support, replace the index URL with the appropriate CUDA version, e.g.:*
`--index-url https://download.pytorch.org/whl/cu121`

### 4. Install OpenMMLab packages

```bash
# Install setuptools/wheel first (required for some builds)
pip install setuptools wheel

# Install mmengine and mmcv-lite (compatible version)
pip install mmengine
pip install "mmcv-lite>=2.0.0rc4,<2.2.0"
```

**Note:** `mmcv-lite` is preferred over `mmcv` (full) when CUDA ops are not needed. The full `mmcv` package requires CUDA and a source build which may fail on Python 3.13 due to the removal of `pkg_resources`.

### 5. Install MMAction2 and video decoder

```bash
pip install mmaction2 decord
```

### 6. Verify installation

```bash
python3 -c "import torch; import decord; import mmcv; import mmaction; print('All imports OK'); print('mmcv version:', mmcv.__version__); print('mmaction version:', mmaction.__version__)"
```

Expected output:

```text
All imports OK
mmcv version: 2.1.0
mmaction version: 1.2.0
```

## Version Compatibility Notes

| Component | Installed Version |
| ----------- | ------------------- |
| Python | 3.13.11 |
| torch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| mmcv-lite | 2.1.0 |
| mmengine | 0.10.7 |
| mmaction2 | 1.2.0 |
| decord | 0.6.0 |

## Troubleshooting

### `mmcv` build fails with `ModuleNotFoundError: No module named 'pkg_resources'`

- Install `setuptools` and `wheel` before building mmcv
- Or switch to `mmcv-lite` which has a pre-built wheel for Python 3.13

### `mmcv-lite` version incompatibility with `mmaction2`

- mmaction2 requires `mmcv>=2.0.0rc4, <2.2.0`
- Pin the install: `"mmcv-lite>=2.0.0rc4,<2.2.0"`
- mmcv-lite 2.1.0 satisfies this constraint

### Decord installation

- `pip install decord` installs the CPU-only version
- For GPU-accelerated decoding, additional setup may be needed
