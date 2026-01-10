#!/usr/bin/env python3
"""
CLI entry point for downloading GGUF models (l3m-download command).

This is a thin wrapper around the download script.
"""

from l3m_backend.scripts.download_gguf import main

if __name__ == "__main__":
    main()
