# l3m-chat

> Interactive chat REPL with LLM tool calling support.
> More information: <https://github.com/mo/l3m-backend>.

- Start chat with auto-detected model:

`l3m-chat`

- Start chat with a specific model file:

`l3m-chat {{model.gguf}}`

- List available models in ~/.l3m/models/:

`l3m-chat --list`

- Start chat with custom context size:

`l3m-chat --ctx {{8192}}`

- Start chat with specific GPU layers:

`l3m-chat --gpu {{-1}}`

- Start chat in verbose mode:

`l3m-chat -v`

- Slash commands (pure, don't modify history):

`/tools, /history, /clear, /undo, /context, /model, /config, /quit`

- Session commands (manage persistent conversations):

`/session, /sessions, /session-save, /session-load, /session-title, /session-tag`

- Shell command (output not saved to history):

`!{{ls -la}}`

- Magic commands (adds input/output to conversation history):

`%!{{cmd}}, %tool {{name}}, %load {{file}}, %time, %save {{file}}, %edit-response`
