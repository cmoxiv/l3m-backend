# Installation Guide

## Requirements

- Python 3.10, 3.11, or 3.12
- `llama-cpp-python` with appropriate GPU backend

## Basic Installation

```bash
pip install git+https://github.com/your-repo/l3m-backend.git
```

Or for development:

```bash
git clone https://github.com/your-repo/l3m-backend.git
cd l3m-backend
pip install -e '.[dev,llm]'
```

## GPU Acceleration

### macOS (Metal)

For GPU acceleration on Apple Silicon (M1/M2/M3) or AMD GPUs on macOS:

**Option 1: Pre-built wheel (recommended)**

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

**Option 2: Build from source**

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Linux/Windows (CUDA)

For NVIDIA GPU acceleration:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### CPU Only

If you don't need GPU acceleration:

```bash
pip install llama-cpp-python
```

## Verifying GPU Acceleration

### 1. Check at startup

When starting `l3m-chat`, the backend is shown:

```
Backend: Metal (all layers)
```

### 2. Use the `/model` command

In the REPL, type `/model` to see:

```
Model: /path/to/model.gguf
Context: 32768
Backend: Metal
GPU layers: all
```

### 3. Verbose mode

Run with `--verbose` to see llama.cpp initialization:

```bash
l3m-chat --verbose
```

Look for Metal initialization messages:
```
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M2
ggml_metal_init: using Metal
```

## GPU Layer Configuration

Control how many model layers run on GPU:

| Option | Effect |
|--------|--------|
| `--gpu -1` | All layers on GPU (default) |
| `--gpu 0` | CPU only |
| `--gpu N` | N layers on GPU |

Example:
```bash
l3m-chat --gpu 32  # Load 32 layers on GPU
```

Or set as default in config:
```bash
l3m-chat --set-config gpu=-1
```

## Troubleshooting

### Metal not working on macOS

1. Ensure you installed with Metal support (see above)
2. Check macOS version is 11.0 or later
3. Run with `--verbose` to see if Metal initializes

### Low GPU memory

Reduce context size or GPU layers:
```bash
l3m-chat --ctx 8192 --gpu 16
```

### Slow startup

First run may be slow due to Metal shader compilation. Subsequent runs will be faster.
