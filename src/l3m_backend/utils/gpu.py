"""GPU detection utilities for llama-cpp-python."""

import platform
from typing import Any


def get_gpu_info(engine_or_llm: Any) -> dict:
    """Get GPU/Metal information from a ChatEngine or Llama instance.

    Args:
        engine_or_llm: A ChatEngine or llama_cpp.Llama instance.

    Returns:
        dict with keys:
            - backend: "Metal", "CUDA", or "CPU"
            - gpu_layers: Number of GPU layers configured
            - is_metal: True if using Metal on macOS
    """
    # Try to get gpu_layers from various sources
    gpu_layers = None

    # Check if it's a ChatEngine (has _n_gpu_layers)
    if hasattr(engine_or_llm, "_n_gpu_layers"):
        gpu_layers = engine_or_llm._n_gpu_layers
    # Check if it's a Llama instance with n_gpu_layers
    elif hasattr(engine_or_llm, "n_gpu_layers"):
        gpu_layers = engine_or_llm.n_gpu_layers
    # Default to 0 (CPU)
    else:
        gpu_layers = 0

    info = {
        "backend": "CPU",
        "gpu_layers": gpu_layers,
        "is_metal": False,
    }

    # n_gpu_layers: -1 means all layers, 0 means CPU only, >0 means N layers
    if gpu_layers == 0:
        return info

    # Check if on macOS - Metal is the GPU backend
    if platform.system() == "Darwin":
        info["backend"] = "Metal"
        info["is_metal"] = True
    else:
        # Non-macOS with GPU layers - likely CUDA or other
        info["backend"] = "CUDA"

    return info


def format_gpu_status(engine_or_llm: Any) -> str:
    """Format GPU status for display.

    Args:
        engine_or_llm: A ChatEngine or llama_cpp.Llama instance.

    Returns:
        Human-readable string describing GPU status.
    """
    info = get_gpu_info(engine_or_llm)

    layers = info["gpu_layers"]
    layers_str = "all" if layers == -1 else str(layers)

    if info["is_metal"]:
        return f"Metal ({layers_str} layers)"
    elif info["backend"] == "CUDA":
        return f"CUDA ({layers_str} layers)"
    else:
        return "CPU"
