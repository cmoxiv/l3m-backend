#!/usr/bin/env bash
# Test prompt size
# Usage: ./scripts/diagnose-prompt-size.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$HOME/.venvs/l3m/bin/python" ]]; then
    PYTHON="$HOME/.venvs/l3m/bin/python"
elif [[ -f "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

exec "$PYTHON" -m l3m_backend.scripts.diagnose_prompt_size "$@"
