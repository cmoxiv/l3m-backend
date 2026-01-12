# l3m-init

> Initialize l3m-backend configuration directories and settings.
> More information: <https://github.com/mo/l3m-backend>.

- Initialize with default settings (adds missing config keys, preserves existing values):

`l3m-init`

- Initialize and overwrite existing config completely:

`l3m-init --force`

- Initialize quietly (no output):

`l3m-init --quiet`

- Created directories and files:

`~/.l3m/              # Main configuration directory`
`~/.l3m/config.json   # Configuration file`
`~/.l3m/models/       # GGUF model storage`
`~/.l3m/sessions/     # Chat session files`
`~/.l3m/tools/        # User-defined tools`
`~/.l3m/commands/     # User-defined / commands`
`~/.l3m/magic/        # User-defined % magic commands`
