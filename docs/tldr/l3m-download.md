# l3m-download

> Download GGUF models from Hugging Face for use with l3m-chat.
> More information: <https://github.com/mo/l3m-backend>.

- Download a model using a preset:

`l3m-download --preset {{llama3.2-1b}}`

- List available presets:

`l3m-download --list-presets`

- Download to a custom directory:

`l3m-download --preset {{llama3.2-1b}} --output {{/path/to/models}}`

- Download from a Hugging Face repo directly:

`l3m-download --repo {{organization/model-name}} --file {{model.Q4_K_M.gguf}}`
