#!/usr/bin/env python3
"""
Download GGUF model files from Hugging Face or Ollama.

Models are saved to ~/.l3m/models/ by default.

Usage:
    # List available GGUF files in a Hugging Face repo
    l3m-download --list hf:bartowski/Llama-3.2-3B-Instruct-GGUF

    # Download from Hugging Face
    l3m-download hf:bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf

    # Download from Ollama (uses ollama CLI if available)
    l3m-download ollama:llama3.2

    # Use a preset
    l3m-download --preset llama3.2-3b

    # Search Ollama models
    l3m-download --search llama

Sources:
    hf:<repo>       - Hugging Face (e.g., hf:bartowski/Llama-3.2-3B-Instruct-GGUF)
    ollama:<model>  - Ollama registry (e.g., ollama:llama3.2)

Popular Hugging Face repos with GGUF models:
    - bartowski/Llama-3.2-3B-Instruct-GGUF
    - bartowski/Llama-3.2-1B-Instruct-GGUF
    - bartowski/Mistral-7B-Instruct-v0.3-GGUF
    - TheBloke/Llama-2-7B-Chat-GGUF
    - QuantFactory/granite-3.1-8b-instruct-GGUF
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Popular model presets (source, repo/model, filename, description)
PRESETS = {
    # Hugging Face presets
    "llama3.2-3b": (
        "hf",
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "Llama 3.2 3B Instruct (Q4_K_M, ~2GB)",
    ),
    "llama3.2-1b": (
        "hf",
        "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "Llama 3.2 1B Instruct (Q4_K_M, ~0.8GB)",
    ),
    "mistral-7b": (
        "hf",
        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "Mistral 7B Instruct v0.3 (Q4_K_M, ~4.4GB)",
    ),
    "granite-8b": (
        "hf",
        "QuantFactory/granite-3.1-8b-instruct-GGUF",
        "granite-3.1-8b-instruct.Q4_K_M.gguf",
        "IBM Granite 3.1 8B Instruct (Q4_K_M, ~5GB)",
    ),
    "qwen2.5-3b": (
        "hf",
        "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "Qwen 2.5 3B Instruct (Q4_K_M, ~2GB)",
    ),
    "phi3-mini": (
        "hf",
        "bartowski/Phi-3.5-mini-instruct-GGUF",
        "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "Microsoft Phi 3.5 Mini (Q4_K_M, ~2.4GB)",
    ),
    # Ollama presets
    "ollama-llama3.2": (
        "ollama",
        "llama3.2",
        None,
        "Llama 3.2 via Ollama",
    ),
    "ollama-llama3.2:1b": (
        "ollama",
        "llama3.2:1b",
        None,
        "Llama 3.2 1B via Ollama",
    ),
    "ollama-mistral": (
        "ollama",
        "mistral",
        None,
        "Mistral 7B via Ollama",
    ),
    "ollama-qwen2.5": (
        "ollama",
        "qwen2.5",
        None,
        "Qwen 2.5 via Ollama",
    ),
    "ollama-phi3": (
        "ollama",
        "phi3",
        None,
        "Microsoft Phi 3 via Ollama",
    ),
}

HF_API_BASE = "https://huggingface.co/api/models/"
HF_DOWNLOAD_BASE = "https://huggingface.co/"
OLLAMA_REGISTRY = "https://registry.ollama.ai"

# Default models directory
DEFAULT_MODELS_DIR = Path.home() / ".l3m" / "models"
OLLAMA_MODELS_DIR = Path.home() / ".ollama" / "models"


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def has_ollama_cli() -> bool:
    """Check if ollama CLI is available."""
    return shutil.which("ollama") is not None


def parse_source(source: str) -> tuple[str, str]:
    """Parse source string into (type, identifier).

    Examples:
        hf:bartowski/Llama-3.2-3B-Instruct-GGUF -> ("hf", "bartowski/Llama-3.2-3B-Instruct-GGUF")
        ollama:llama3.2 -> ("ollama", "llama3.2")
        bartowski/Llama-3.2-3B-Instruct-GGUF -> ("hf", "bartowski/Llama-3.2-3B-Instruct-GGUF")
    """
    if source.startswith("hf:"):
        return ("hf", source[3:])
    elif source.startswith("ollama:"):
        return ("ollama", source[7:])
    elif "/" in source:
        # Assume Hugging Face repo format (org/repo)
        return ("hf", source)
    else:
        # Assume Ollama model name
        return ("ollama", source)


# =============================================================================
# Hugging Face Functions
# =============================================================================

def list_gguf_files(repo: str) -> list[dict]:
    """List GGUF files in a Hugging Face repo."""
    url = f"{HF_API_BASE}{repo}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching repo info: {e}")
        return []

    siblings = data.get("siblings", [])
    gguf_files = []
    for file_info in siblings:
        filename = file_info.get("rfilename", "")
        if filename.endswith(".gguf"):
            gguf_files.append({
                "filename": filename,
                "size": file_info.get("size", 0),
            })

    return sorted(gguf_files, key=lambda x: x["filename"])


def download_file(url: str, output_path: Path, expected_size: int = 0) -> bool:
    """Download file with progress bar."""
    print(f"Downloading: {url}")
    print(f"To: {output_path}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error starting download: {e}")
        return False

    total_size = int(response.headers.get("content-length", expected_size))
    downloaded = 0
    chunk_size = 8192

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_len = 40
                        filled = int(bar_len * downloaded / total_size)
                        bar = "=" * filled + "-" * (bar_len - filled)
                        print(
                            f"\r[{bar}] {percent:.1f}% ({format_size(downloaded)}/{format_size(total_size)})",
                            end="",
                            flush=True,
                        )
                    else:
                        print(f"\rDownloaded: {format_size(downloaded)}", end="", flush=True)
        print()  # newline after progress
        return True
    except (IOError, KeyboardInterrupt) as e:
        print(f"\nDownload failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_from_hf(repo: str, filename: str, output_dir: Path) -> bool:
    """Download a GGUF file from Hugging Face."""
    # Get file info first
    files = list_gguf_files(repo)
    file_info = next((f for f in files if f["filename"] == filename), None)

    if not file_info:
        print(f"File not found: {filename}")
        print(f"\nAvailable GGUF files in {repo}:")
        for f in files:
            print(f"  - {f['filename']} ({format_size(f['size'])})")
        return False

    # Build download URL
    download_url = f"{HF_DOWNLOAD_BASE}{repo}/resolve/main/{filename}"
    output_path = output_dir / filename

    if output_path.exists():
        existing_size = output_path.stat().st_size
        # If API doesn't report size (returns 0), assume existing file is complete
        if file_info["size"] == 0 or existing_size == file_info["size"]:
            print(f"File already exists: {output_path} ({format_size(existing_size)})")
            return True
        else:
            print(f"Partial download detected ({format_size(existing_size)} / {format_size(file_info['size'])}), re-downloading...")

    print(f"\nFile: {filename}")
    print(f"Size: {format_size(file_info['size'])}")
    print()

    success = download_file(download_url, output_path, file_info["size"])
    if success:
        print(f"\nDownload complete: {output_path}")

        # Verify size
        actual_size = output_path.stat().st_size
        if actual_size != file_info["size"]:
            print(f"Warning: Size mismatch (expected {file_info['size']}, got {actual_size})")
        else:
            print(f"Size verified: {format_size(actual_size)}")

    return success


# =============================================================================
# Ollama Functions
# =============================================================================

def ollama_pull(model: str) -> bool:
    """Pull a model using ollama CLI."""
    if not has_ollama_cli():
        print("Error: ollama CLI not found.")
        print("Install Ollama from: https://ollama.ai")
        return False

    print(f"Pulling model via Ollama: {model}")
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            check=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model: {e}")
        return False
    except FileNotFoundError:
        print("Error: ollama command not found")
        return False


def ollama_list() -> list[dict]:
    """List locally available Ollama models."""
    if not has_ollama_cli():
        return []

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        models = []
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:  # Skip header
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    models.append({
                        "name": parts[0],
                        "size": parts[2] if len(parts) > 2 else "?",
                        "modified": parts[3] if len(parts) > 3 else "?",
                    })
        return models
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def ollama_show(model: str) -> Optional[dict]:
    """Get details about an Ollama model."""
    if not has_ollama_cli():
        return None

    try:
        result = subprocess.run(
            ["ollama", "show", model, "--modelfile"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"modelfile": result.stdout}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def search_ollama_models(query: str) -> list[dict]:
    """Search Ollama library for models (scrapes ollama.ai/library)."""
    # Note: Ollama doesn't have a public search API, so we provide popular models
    popular_models = [
        {"name": "llama3.2", "description": "Meta Llama 3.2 (1B, 3B)", "size": "1.3GB-2GB"},
        {"name": "llama3.2:1b", "description": "Meta Llama 3.2 1B", "size": "1.3GB"},
        {"name": "llama3.2:3b", "description": "Meta Llama 3.2 3B", "size": "2GB"},
        {"name": "llama3.1", "description": "Meta Llama 3.1 (8B, 70B, 405B)", "size": "4.7GB+"},
        {"name": "mistral", "description": "Mistral 7B", "size": "4.1GB"},
        {"name": "mixtral", "description": "Mixtral 8x7B MoE", "size": "26GB"},
        {"name": "phi3", "description": "Microsoft Phi-3 (mini, small, medium)", "size": "2.2GB+"},
        {"name": "qwen2.5", "description": "Alibaba Qwen 2.5 (0.5B-72B)", "size": "0.4GB+"},
        {"name": "gemma2", "description": "Google Gemma 2 (2B, 9B, 27B)", "size": "1.6GB+"},
        {"name": "codellama", "description": "Meta Code Llama", "size": "3.8GB+"},
        {"name": "deepseek-coder", "description": "DeepSeek Coder", "size": "776MB+"},
        {"name": "starcoder2", "description": "StarCoder2", "size": "1.7GB+"},
        {"name": "granite-code", "description": "IBM Granite Code", "size": "2GB+"},
        {"name": "command-r", "description": "Cohere Command R", "size": "20GB+"},
        {"name": "orca-mini", "description": "Orca Mini", "size": "1.9GB+"},
        {"name": "vicuna", "description": "Vicuna", "size": "3.8GB+"},
        {"name": "neural-chat", "description": "Intel Neural Chat", "size": "4.1GB"},
        {"name": "stablelm2", "description": "Stability AI StableLM 2", "size": "982MB+"},
        {"name": "tinyllama", "description": "TinyLlama 1.1B", "size": "637MB"},
        {"name": "dolphin-phi", "description": "Dolphin Phi-2", "size": "1.6GB"},
    ]

    query_lower = query.lower()
    matches = [m for m in popular_models if query_lower in m["name"].lower() or query_lower in m["description"].lower()]
    return matches


def find_ollama_gguf(model: str) -> Optional[Path]:
    """Find the GGUF file for an Ollama model.

    Ollama stores models as blobs in ~/.ollama/models/blobs/.
    The manifest links digests to layers.
    """
    # Parse model name and tag
    if ":" in model:
        name, tag = model.split(":", 1)
    else:
        name, tag = model, "latest"

    # Find manifest
    manifest_path = OLLAMA_MODELS_DIR / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
        # Find the model layer (largest layer, usually)
        layers = manifest.get("layers", [])
        model_layer = None
        for layer in layers:
            media_type = layer.get("mediaType", "")
            if "model" in media_type:
                model_layer = layer
                break

        if not model_layer:
            # Fallback: find largest layer
            model_layer = max(layers, key=lambda x: x.get("size", 0), default=None)

        if model_layer:
            digest = model_layer.get("digest", "").replace(":", "-")
            blob_path = OLLAMA_MODELS_DIR / "blobs" / digest
            if blob_path.exists():
                return blob_path
    except (json.JSONDecodeError, KeyError):
        pass

    return None


def link_ollama_model(model: str, output_dir: Path) -> Optional[Path]:
    """Create a symlink to an Ollama model's GGUF file."""
    gguf_path = find_ollama_gguf(model)
    if not gguf_path:
        print(f"Could not find GGUF blob for Ollama model: {model}")
        return None

    # Create a symlink with a readable name
    safe_name = model.replace(":", "-").replace("/", "-")
    link_name = f"ollama-{safe_name}.gguf"
    link_path = output_dir / link_name

    output_dir.mkdir(parents=True, exist_ok=True)

    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()

    link_path.symlink_to(gguf_path)
    print(f"Created symlink: {link_path} -> {gguf_path}")
    return link_path


def download_from_ollama(model: str, output_dir: Path, link: bool = True) -> bool:
    """Download a model from Ollama and optionally create a symlink."""
    # First, pull the model
    success = ollama_pull(model)
    if not success:
        return False

    # Create symlink to the GGUF file
    if link:
        link_path = link_ollama_model(model, output_dir)
        if link_path:
            print(f"\nModel available at: {link_path}")
            return True
        else:
            print(f"\nModel pulled but could not create symlink.")
            print(f"Model files are in: {OLLAMA_MODELS_DIR}")
            return True

    return True


# =============================================================================
# CLI Functions
# =============================================================================

def print_presets():
    """Print available presets."""
    print("\nAvailable presets:")
    print("-" * 70)
    print("\nHugging Face presets:")
    for name, (source, repo, filename, desc) in PRESETS.items():
        if source == "hf":
            print(f"  {name:<20} - {desc}")

    print("\nOllama presets:")
    for name, (source, repo, filename, desc) in PRESETS.items():
        if source == "ollama":
            print(f"  {name:<20} - {desc}")

    print()
    print("Usage: l3m-download --preset <name>")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download GGUF models from Hugging Face or Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sources:
    hf:<repo>       - Hugging Face (e.g., hf:bartowski/Llama-3.2-3B-Instruct-GGUF)
    ollama:<model>  - Ollama registry (e.g., ollama:llama3.2)

    If no prefix given:
      - Contains '/' -> treated as Hugging Face repo
      - Otherwise -> treated as Ollama model

Examples:
    # List files in a Hugging Face repo
    l3m-download --list hf:bartowski/Llama-3.2-3B-Instruct-GGUF

    # Download from Hugging Face
    l3m-download hf:bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf

    # Download from Ollama
    l3m-download ollama:llama3.2

    # Use preset
    l3m-download --preset llama3.2-3b

    # Search Ollama models
    l3m-download --search llama

    # List local Ollama models
    l3m-download --ollama-list

    # Show all presets
    l3m-download --presets
        """,
    )
    parser.add_argument("source", nargs="?", help="Source (hf:<repo> or ollama:<model>)")
    parser.add_argument("filename", nargs="?", help="GGUF filename (for Hugging Face)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help=f"Output directory (default: {DEFAULT_MODELS_DIR})")
    parser.add_argument("--list", action="store_true", help="List GGUF files in Hugging Face repo")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), help="Use a model preset")
    parser.add_argument("--presets", action="store_true", help="Show available presets")
    parser.add_argument("--make-default", action="store_true", help="Create model.gguf symlink (makes this the default model)")
    parser.add_argument("--search", type=str, metavar="QUERY", help="Search Ollama models")
    parser.add_argument("--ollama-list", action="store_true", help="List local Ollama models")
    parser.add_argument("--no-link", action="store_true", help="Don't create symlink for Ollama models")
    args = parser.parse_args()

    if not HAS_REQUESTS:
        print("Error: requests library required.")
        print("Install with: pip install requests")
        sys.exit(1)

    # Show presets
    if args.presets:
        print_presets()
        return

    # Search Ollama models
    if args.search:
        results = search_ollama_models(args.search)
        if results:
            print(f"\nOllama models matching '{args.search}':")
            print("-" * 60)
            for m in results:
                print(f"  {m['name']:<20} {m['description']:<30} ({m['size']})")
            print()
            print("Download with: l3m-download ollama:<model>")
        else:
            print(f"No models found matching '{args.search}'")
        return

    # List local Ollama models
    if args.ollama_list:
        models = ollama_list()
        if models:
            print("\nLocal Ollama models:")
            print("-" * 60)
            for m in models:
                print(f"  {m['name']:<30} {m['size']:<10}")
        else:
            print("No local Ollama models found.")
            if not has_ollama_cli():
                print("Ollama CLI not installed. Install from: https://ollama.ai")
        return

    # Handle preset
    if args.preset:
        source_type, repo_or_model, filename, desc = PRESETS[args.preset]
        print(f"Preset: {args.preset}")
        print(f"Description: {desc}")
        if source_type == "hf":
            args.source = f"hf:{repo_or_model}"
            args.filename = filename
        else:
            args.source = f"ollama:{repo_or_model}"

    # Validate args
    if not args.source:
        parser.print_help()
        print("\n" + "=" * 60)
        print_presets()
        return

    output_dir = Path(args.output) if args.output else DEFAULT_MODELS_DIR

    # Parse source
    source_type, identifier = parse_source(args.source)

    # List mode (Hugging Face only)
    if args.list:
        if source_type != "hf":
            print("--list is only supported for Hugging Face repos")
            print("Use --ollama-list to list local Ollama models")
            sys.exit(1)

        print(f"\nGGUF files in {identifier}:")
        print("-" * 60)
        files = list_gguf_files(identifier)
        if not files:
            print("  No GGUF files found (or repo doesn't exist)")
        else:
            for f in files:
                print(f"  {f['filename']:<50} {format_size(f['size']):>10}")
        print()
        return

    # Download based on source type
    if source_type == "hf":
        if not args.filename:
            print("Error: filename required for Hugging Face download")
            print(f"\nUse --list to see available files:")
            print(f"  l3m-download --list hf:{identifier}")
            sys.exit(1)

        success = download_from_hf(identifier, args.filename, output_dir)

        if success and args.make_default:
            symlink_path = output_dir / "model.gguf"
            target_path = output_dir / args.filename
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(target_path.name)
            print(f"Set as default: {symlink_path} -> {target_path.name}")

    elif source_type == "ollama":
        success = download_from_ollama(identifier, output_dir, link=not args.no_link)

    else:
        print(f"Unknown source type: {source_type}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
