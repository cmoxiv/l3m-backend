"""Model command - show/switch model."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from l3m_backend.cli.commands.registry import command_registry

if TYPE_CHECKING:
    from l3m_backend.engine import ChatEngine
    from l3m_backend.session import SessionManager

# Models directory
MODEL_DIR = Path.home() / ".l3m" / "models"


@command_registry.register("model", "Show/switch model", usage="/model [name]", has_args=True)
def cmd_model(engine: "ChatEngine", session_mgr: "SessionManager | None", args: str):
    """Show model info or switch model."""
    from l3m_backend.utils.gpu import get_gpu_info

    if not args:
        # Show current model info
        llm = engine.llm
        model_path = getattr(llm, "model_path", "unknown")
        try:
            n_ctx = llm.n_ctx()
        except (AttributeError, TypeError):
            n_ctx = "?"
        gpu_info = get_gpu_info(engine)
        layers = gpu_info["gpu_layers"]
        layers_str = "all" if layers == -1 else str(layers)

        print(f"\nModel: {model_path}")
        print(f"Context: {n_ctx}")
        print(f"Backend: {gpu_info['backend']}")
        print(f"GPU layers: {layers_str}")
        print(f"\nAvailable models in {MODEL_DIR}:")
        if MODEL_DIR.exists():
            for m in sorted(MODEL_DIR.glob("*.gguf")):
                size_mb = m.stat().st_size / (1024 * 1024)
                print(f"  {m.name:<50} {size_mb:>8.1f} MB")
        print("\nUsage: /model <name> to switch models\n")
    else:
        # Switch to specified model
        model_name = args.strip()
        new_model_path = MODEL_DIR / model_name
        if not new_model_path.exists():
            # Try as full path
            new_model_path = Path(model_name)
            if not new_model_path.exists():
                print(f"Model not found: {model_name}")
                print(f"Looked in: {MODEL_DIR / model_name}")
                return

        print(f"Switching to {new_model_path.name}...")
        try:
            engine.switch_model(str(new_model_path))
            from l3m_backend.utils.gpu import get_gpu_info
            gpu_info = get_gpu_info(engine)
            print(f"Switched to: {new_model_path.name}")
            print(f"Context: {engine.llm.n_ctx()}")
            print(f"Backend: {gpu_info['backend']}\n")
        except Exception as e:
            print(f"Failed to switch model: {e}\n")
