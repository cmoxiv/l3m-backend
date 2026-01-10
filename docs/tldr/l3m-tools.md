# l3m-tools

> Manage and inspect LLM tools in the l3m-backend registry.
> More information: <https://github.com/mo/l3m-backend>.

- List all available tools:

`l3m-tools list`

- List tools with full descriptions:

`l3m-tools list -v`

- Show OpenAI-compatible tool schema (JSON):

`l3m-tools schema`

- Show schema for a specific tool:

`l3m-tools schema {{tool_name}}`

- Show detailed info about a tool:

`l3m-tools info {{tool_name}}`

- Create a new user tool (scaffold):

`l3m-tools create {{my_tool}}`

- Create a wrapper tool for an external command:

`l3m-tools create {{my_tool}} --wrapper`

- Create tool, overwriting if exists:

`l3m-tools create {{my_tool}} --force`
