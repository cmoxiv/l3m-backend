# Standups

## 2026-01-10 09:45

**Branch:** main

### Completed
- Fixed session persistence bugs:
  - Symlink not updated on save (timing issue in `manager.py:203-209`)
  - Ctrl+D exit didn't save session (both REPLs)
  - Improved Ctrl+D exit UX with confirmation flow
- Tool calling module (`toolcall.py`) completed with support for 6 formats (CONTRACT, OPENAI, HERMES, LLAMA3, MISTRAL, NATIVE)
- E2E test suite for Hermes 8B (`scripts/e2e_test_hermes.py`) - 5/5 tests passing
- E2E test script for native tool calling (`scripts/e2e_test_native.py`)
- Minimal contract optimization reducing token usage (2228 -> 634 chars) with balanced examples
- All 25 session tests passing

### In Progress
- Uncommitted changes across entire `src/` directory, tests, docs, and scripts
- Native tool calling investigation (documented limitations with chatml-function-calling format)

### Blockers
- None

### Next Steps
- Commit current changes (large uncommitted delta)
- Add integration tests for REPL exit scenarios
- Investigate llama-cpp-python native tool calling for better model support
- Test with larger models (70B) for improved instruction following
- Consider CI integration for E2E tests (requires GPU)

## 2026-01-11 09:00

**Branch:** main

### Completed
- Fixed MCP tool registration issues:
  - Params model issue: MCP tools with `**kwargs` generated empty schemas, dropping arguments on validation
  - Name normalization issue: Tools with dots (e.g., `mcp.test.score`) weren't matching normalized lookups
  - Fix implemented in `src/l3m_backend/mcp/client/registry_adapter.py`
- MCP HTTP server example created (`examples/mcp_server_http/`)
- Man page documentation for l3m-chat (`docs/man/l3m-chat.1`)
- Frontend test suite (`tests/test_frontend.py`)

### In Progress
- Large uncommitted delta across entire codebase (src/, tests/, docs/, scripts/, examples/)
- README updates

### Blockers
- None

### Next Steps
- Commit current changes to main
- Add MCP integration tests
- Test MCP client with external servers
- Consider adding HTTP transport option to MCP client

## Session Wrap-up

**Date:** 2026-01-10

### What Got Done
- Fixed session persistence bugs: symlink update timing issue, Ctrl+D exit session saving, improved Ctrl+D exit UX with confirmation flow
- Completed tool calling module (`toolcall.py`) with support for 6 formats (CONTRACT, OPENAI, HERMES, LLAMA3, MISTRAL, NATIVE)
- Built E2E test suite for Hermes 8B with all 5 tests passing
- Created E2E test script for native tool calling
- Optimized minimal contract, reducing token usage from 2228 to 634 characters
- All 25 session tests passing
- Investigated native tool calling and documented limitations with chatml-function-calling format

### Summary
This session focused on stabilizing session persistence and expanding tool calling capabilities. Major wins include fixing critical save/exit bugs, completing a multi-format tool calling module, and significantly reducing token overhead through contract optimization.

## Session Wrap-up

**Date:** 2026-01-12

### What Got Done
- Fixed similarity graph buffer size error (20.55 GiB allocation) by setting embedding model context to 512 tokens
- Added configurable stop tokens to prevent chatty model output (`<|end|>`, `<|eot_id|>`, `<|im_end|>`)
- Implemented true token-by-token streaming for final responses
- Made `l3m-init` preserve existing config values (only adds missing keys)
- Added history trimming on session resume with summarization when context is smaller than history
- Fixed `/clear` command to properly clear history, transcript, and summaries from session
- Created real DuckDuckGo web search tool (Instant Answer API + HTML lite fallback)
- Isolated tools into separate folders: `wikipedia/`, `dictionary/`, `unit_convert/`, `currency/`, `web_search/`
- Renamed CLI options for consistency: `--config-set`, `--config-del`, `--config-save-default`, `--config-make-default`
- Added auto-completion script installation to `l3m-init` (`~/.l3m/completions/`)
- Updated default system prompt with clear tool usage guidelines
- Fixed Ctrl+C handling to show "You:" prompt immediately
- **Implemented dynamic priming mechanism** with progressive examples (simple → multi-request → chaining with planning)
- Created `~/.l3m/priming.yaml` for user-configurable tool usage examples

### Summary
This session focused on UX improvements and implementing a dynamic priming system for tool contracts. Key achievements include fixing streaming behavior, improving session management, isolating tools into modular folders, and creating a sophisticated priming mechanism that generates progressive examples based on available tools to teach the LLM effective tool usage patterns.
