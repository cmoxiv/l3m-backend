#!/usr/bin/env python3
"""
CLI completion script generator for l3m commands.

Usage:
    # Generate and source bash completion
    source <(l3m-completion bash)

    # Add to .bashrc for persistent completion
    echo 'source <(l3m-completion bash)' >> ~/.bashrc
"""

import argparse
import signal
import sys
from pathlib import Path

DEFAULT_MODELS_DIR = Path.home() / ".l3m" / "models"

BASH_COMPLETION = r'''
# l3m-chat bash completion
_l3m_chat_completions() {
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local models_dir="$HOME/.l3m/models"
    local sessions_dir="$HOME/.l3m/sessions"
    local opts="--list -l --ctx --gpu --verbose -v --simple --session -s --resume -r --incognito --list-sessions --search-sessions --config --config-set --config-del --config-save-default --config-make-default --help --no-warmup --summary-ctx --transcript-ctx"
    local config_keys="default_model ctx gpu verbose simple incognito auto_resume system_prompt temperature max_tokens"

    # Handle option arguments
    case "$prev" in
        --ctx|--gpu|--summary-ctx|--transcript-ctx)
            # Numeric arguments - no completion
            return 0
            ;;
        --session|-s)
            # Session completion: show title-shortid format
            if [[ -d "$sessions_dir" ]]; then
                # Extract title-id from filename: YYMMDD-HHMM-YYMMDD-HHMM-tag-title-id.json -> title-id
                local sessions=$(find "$sessions_dir" -maxdepth 1 -name "*.json" -exec basename {} .json \; 2>/dev/null | cut -d- -f6-)
                COMPREPLY=($(compgen -W "$sessions" -- "$cur"))
            fi
            return 0
            ;;
        --config-set)
            # Config key= completion
            local keyvals=""
            for key in $config_keys; do
                keyvals="$keyvals ${key}="
            done
            COMPREPLY=($(compgen -W "$keyvals" -- "$cur"))
            return 0
            ;;
        --config-del)
            COMPREPLY=($(compgen -W "$config_keys" -- "$cur"))
            return 0
            ;;
        --search-sessions)
            # Free text - no completion
            return 0
            ;;
    esac

    # Options completion (when cur starts with -)
    if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "$opts" -- "$cur"))
        return 0
    fi

    # Model completion (positional argument)
    local models=""
    if [[ -d "$models_dir" ]]; then
        models=$(find "$models_dir" -maxdepth 1 -name "*.gguf" -exec basename {} \; 2>/dev/null)
    fi
    if [[ -n "$models" ]]; then
        COMPREPLY=($(compgen -W "$models" -- "$cur"))
    fi
}

# l3m-init bash completion
_l3m_init_completions() {
    local cur
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"

    if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "--force -f --config-save-default --config-make-default --quiet -q --help" -- "$cur"))
    fi
}

# l3m-download bash completion
_l3m_download_completions() {
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    case "$prev" in
        --preset)
            COMPREPLY=($(compgen -W "llama3.2-3b llama3.2-1b mistral-7b granite-8b qwen2.5-3b phi3-mini" -- "$cur"))
            return 0
            ;;
        --output|-o)
            COMPREPLY=($(compgen -d -- "$cur"))
            return 0
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        COMPREPLY=($(compgen -W "--list --preset --presets --output -o --symlink --help" -- "$cur"))
    fi
}

# l3m-tools bash completion
_l3m_tools_completions() {
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Handle --wrap argument
    if [[ "$prev" == "--wrap" ]]; then
        COMPREPLY=($(compgen -c -- "$cur"))
        return 0
    fi

    # Subcommand-specific options
    if [[ "$cur" == -* ]]; then
        case "${COMP_WORDS[1]}" in
            create)
                COMPREPLY=($(compgen -W "--wrap --force --help" -- "$cur"))
                ;;
            list|info)
                COMPREPLY=($(compgen -W "--json --help" -- "$cur"))
                ;;
            *)
                COMPREPLY=($(compgen -W "--help" -- "$cur"))
                ;;
        esac
        return 0
    fi

    # Subcommand completion (first argument after l3m-tools)
    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=($(compgen -W "list schema info create" -- "$cur"))
    fi
}

# Register completions
complete -o default -F _l3m_chat_completions l3m-chat
complete -o default -F _l3m_init_completions l3m-init
complete -o default -F _l3m_download_completions l3m-download
complete -o default -F _l3m_tools_completions l3m-tools
'''

ZSH_COMPLETION = r'''
# l3m-chat zsh completion
_l3m_chat() {
    local models_dir="$HOME/.l3m/models"
    local sessions_dir="$HOME/.l3m/sessions"
    local -a models sessions config_keys

    config_keys=(default_model ctx gpu verbose simple incognito auto_resume system_prompt temperature max_tokens)

    if [[ -d "$models_dir" ]]; then
        models=(${models_dir}/*.gguf(N:t))
    fi

    if [[ -d "$sessions_dir" ]]; then
        # Extract short IDs from session filenames
        sessions=($(ls "$sessions_dir"/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/.*-\\([^-]*\\)\\.json$/\\1/'))
    fi

    _arguments \\
        '1:model:->models' \\
        '--list[List available models]' \\
        '-l[List available models]' \\
        '--ctx[Context size]:size:' \\
        '--gpu[GPU layers]:layers:' \\
        '--verbose[Verbose output]' \\
        '-v[Verbose output]' \\
        '--simple[Use simple REPL]' \\
        '--session[Session ID]:session:->sessions' \\
        '-s[Session ID]:session:->sessions' \\
        '--resume[Resume session from CWD]' \\
        '-r[Resume session from CWD]' \\
        '--incognito[Incognito mode]' \\
        '--list-sessions[List sessions]' \\
        '--search-sessions[Search sessions]:query:' \\
        '--config[Show config]' \\
        '--config-set[Set config]:key=value:->setconfig' \\
        '--config-del[Unset config]:key:->unsetconfig' \\
        '--config-save-default[Save current config as default]' \\
        '--config-make-default[Create default from package defaults]' \\
        '--help[Show help]'

    case $state in
        models)
            _values 'model' $models
            _files -g '*.gguf'
            ;;
        sessions)
            _values 'session' $sessions
            ;;
        setconfig)
            local -a keyvals
            for k in $config_keys; do
                keyvals+=("${k}=")
            done
            _values 'config' $keyvals
            ;;
        unsetconfig)
            _values 'config' $config_keys
            ;;
    esac
}

_l3m_init() {
    _arguments \\
        '--force[Overwrite existing config]' \\
        '-f[Overwrite existing config]' \\
        '--config-save-default[Save current config as default]' \\
        '--config-make-default[Create default from package defaults]' \\
        '--quiet[Suppress output]' \\
        '-q[Suppress output]' \\
        '--help[Show help]'
}

_l3m_download() {
    local -a presets
    presets=(llama3.2-3b llama3.2-1b mistral-7b granite-8b qwen2.5-3b phi3-mini)

    _arguments \\
        '--list[List GGUF files in repo]' \\
        '--preset[Use model preset]:preset:($presets)' \\
        '--presets[Show available presets]' \\
        '-o[Output directory]:dir:_files -/' \\
        '--output[Output directory]:dir:_files -/' \\
        '--symlink[Create model.gguf symlink]' \\
        '--help[Show help]'
}

_l3m_tools() {
    local -a commands
    commands=(
        'list:List all registered tools'
        'schema:Output OpenAI-style tool schema'
        'info:Show details for a specific tool'
        'create:Create a new user tool'
    )

    _arguments -C \\
        '1:command:->command' \\
        '*::arg:->args'

    case $state in
        command)
            _describe 'command' commands
            ;;
        args)
            case $words[1] in
                list)
                    _arguments '--json[Output as JSON]' '--help[Show help]'
                    ;;
                info)
                    _arguments '1:tool:' '--json[Output as JSON]' '--help[Show help]'
                    ;;
                create)
                    _arguments \\
                        '1:name:' \\
                        '--wrap[Wrap shell command]:command:_command_names' \\
                        '--force[Overwrite existing tool]' \\
                        '--help[Show help]'
                    ;;
                schema)
                    _arguments '--help[Show help]'
                    ;;
            esac
            ;;
    esac
}

compdef _l3m_chat l3m-chat
compdef _l3m_init l3m-init
compdef _l3m_download l3m-download
compdef _l3m_tools l3m-tools
'''


DEFAULT_SESSIONS_DIR = Path.home() / ".l3m" / "sessions"

# Config keys for completion
CONFIG_KEYS = [
    "default_model",
    "ctx",
    "gpu",
    "verbose",
    "simple",
    "incognito",
    "auto_resume",
    "system_prompt",
    "temperature",
    "max_tokens",
]


def list_models():
    """List model names for completion."""
    if not DEFAULT_MODELS_DIR.exists():
        return []
    return [p.name for p in DEFAULT_MODELS_DIR.glob("*.gguf")]


def list_sessions():
    """List session short IDs for completion."""
    if not DEFAULT_SESSIONS_DIR.exists():
        return []
    sessions = []
    for p in DEFAULT_SESSIONS_DIR.glob("*.json"):
        # Extract short ID from filename: {timestamp}-{tag}-{prompt}-{shortid}.json
        parts = p.stem.rsplit("-", 1)
        if len(parts) == 2:
            sessions.append(parts[1])
        else:
            sessions.append(p.stem)
    return sessions


def main():
    """Generate shell completion scripts."""
    # Handle broken pipe gracefully (e.g., when used with process substitution)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    parser = argparse.ArgumentParser(
        description="Generate shell completion scripts for l3m commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
    # Source bash completion
    source <(l3m-completion bash)

    # Add to .bashrc
    echo 'source <(l3m-completion bash)' >> ~/.bashrc

    # Source zsh completion
    source <(l3m-completion zsh)

    # List models (for custom completion)
    l3m-completion models

    # List sessions (for custom completion)
    l3m-completion sessions

    # List config keys (for custom completion)
    l3m-completion config-keys
        """,
    )
    parser.add_argument(
        "shell",
        choices=["bash", "zsh", "models", "sessions", "config-keys"],
        help="Shell type or helper command for custom completion",
    )
    args = parser.parse_args()

    if args.shell == "bash":
        print(BASH_COMPLETION)
    elif args.shell == "zsh":
        print(ZSH_COMPLETION)
    elif args.shell == "models":
        for model in list_models():
            print(model)
    elif args.shell == "sessions":
        for session in list_sessions():
            print(session)
    elif args.shell == "config-keys":
        for key in CONFIG_KEYS:
            print(key)


if __name__ == "__main__":
    main()
