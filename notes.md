# Notes

## Note 1 — DONE

**Raw:**
> `/session-title` pure command not taking into consideration the entire conversation.

**Revised:**
The `/session-title` command generates titles based only on the immediate context, ignoring the full conversation history. It should analyze the entire session to produce a more accurate title.

---

## Note 2 — DONE

**Raw:**
> add another context window to load the summarise conversations when loading saved sessions with a swithc --summary-ctx (0 for disabled, >0 for size, -1 for add to system default 4096)

**Revised:**
Add a `--summary-ctx` flag for loading saved sessions with conversation summaries:
- `0` — Disabled (no summary context)
- `>0` — Use specified context size for summaries
- `-1` — Add summary context on top of the system default (4096 tokens)

This allows summarized conversation history to be loaded into a separate context window when resuming sessions.

---

## Note 3 — DONE

**Raw:**
> add a third context window to load the transcript when loading saved sessions with a switch --transcript-ctx (0 for disabled, >0 for size, -1 for add to system, default is 4096)

**Revised:**
Add a `--transcript-ctx` flag for loading saved sessions with full transcripts:
- `0` — Disabled (no transcript context)
- `>0` — Use specified context size for transcripts
- `-1` — Add transcript context on top of the system default
- Default: 4096 tokens

This creates a third context window dedicated to loading raw conversation transcripts when resuming sessions.

---

## Note 4 — DONE

**Raw:**
> make `/model` able to change model with auto-completion of downloaded models.

**Revised:**
Enhance the `/model` command to support switching models at runtime. Add auto-completion for available models in `~/.l3m/models/` so users can easily select from downloaded GGUF files.

---

## Note 5 — DONE

**Raw:**
> Add a warmup phase when loading a session. The warmup phase, runs the stored summaries and transcripts through the model and asks for summary, title, and completion all of which are not added to history or change anything. It is just to build the KV cache.

**Revised:**
Add a warmup phase when loading saved sessions:
1. Feed stored summaries and transcripts through the model
2. Request summary, title, and completion outputs
3. Discard all outputs (don't add to history or modify state)
4. Purpose: Pre-populate the KV cache for faster subsequent inference

This is a silent preprocessing step that primes the model without affecting the session state.

---

## Note 6

**Raw:**
> Add RAG support with %attach magic command, persistant through session file with the file path, detect if file was changed, when file is attached go over each chunk of text and generate a question that matches the text chunk. We also need a another managed context for the questions and answers.

**Revised:**
Add RAG (Retrieval-Augmented Generation) support:
1. **`%attach` magic command** — Attach files to the session for context
2. **Session persistence** — Store attached file paths in the session file
3. **Change detection** — Detect if attached files have been modified since last load
4. **Question generation** — When a file is attached, chunk the text and generate a matching question for each chunk
5. **Managed context window** — Add a dedicated context partition for storing the generated Q&A pairs from attached files

---

## Note 7 — DONE

**Raw:**
> when doing kv-caching warmup, build a graph network with similarities between
> 1. every msg and every other msg.
> 2. every msg and every summary.
> 3. every summary and every other summary.

**Revised:**
During KV-cache warmup, build a similarity graph network:
1. **Message ↔ Message** — Compute similarity between every message pair
2. **Message ↔ Summary** — Compute similarity between each message and each summary
3. **Summary ↔ Summary** — Compute similarity between every summary pair

This graph can be used for context-aware retrieval, deduplication, or relevance ranking during session loading.

**Implementation:**
- `src/l3m_backend/engine/similarity.py` — SimilarityGraph, LlamaEmbeddingProvider
- `engine.warmup(build_similarity_graph=True)` to build during warmup
- APIs: `get_similar_messages()`, `get_relevant_context()`, `find_duplicates()`, `rank_messages_by_relevance()`

---

## Note 8 — DONE

**Raw:**
> Add full MCP support for l3m. All features of MCP must be implemented.

**Revised:**
Implement full MCP (Model Context Protocol) support for l3m-backend:
- **MCP Server** — Expose l3m tools as an MCP server
- **MCP Client** — Connect to external MCP servers and use their tools
- **Resources** — Support MCP resource types (files, URIs, etc.)
- **Prompts** — Support MCP prompt templates
- **Sampling** — Support MCP sampling requests
- **Transport** — Support stdio and HTTP/SSE transports

All MCP specification features should be implemented for full compatibility.

---

## Note 9 — DONE

**Raw:**
> Add an example MCP server to test l3m's MCP support.

**Revised:**
Create an example MCP server for testing l3m's MCP client implementation:
- Provide sample tools, resources, and prompts
- Serve as a reference implementation and test fixture
- Include in `examples/` or `tests/` directory
- Document how to run and connect to it

---

## Note 10 — DONE

**Raw:**
> Isolate l3m's pure commands to read command functionality from ~/.l3m/commands

**Revised:**
Refactor l3m's pure commands (`/` commands) to be plugin-based:
- Move command implementations out of the REPL code
- Load commands dynamically from `~/.l3m/commands/` directory
- Allow users to add custom `/` commands similar to user tools
- Keep built-in commands but allow overriding or extending

---

## Note 11 — DONE

**Raw:**
> Isolate l3m's magic commands to read magic command functionality from ~/.l3m/magic

**Revised:**
Refactor l3m's magic commands (`%` commands) to be plugin-based:
- Move magic command implementations out of `_magic.py`
- Load magic commands dynamically from `~/.l3m/magic/` directory
- Allow users to add custom `%` commands similar to user tools
- Keep built-in magic commands but allow overriding or extending

---

## Note 12 — DONE

**Raw:**
> Store prompt history at ~/.l3m/prompt_history

**Revised:**
Move prompt history storage to `~/.l3m/prompt_history`:
- Currently stored at `~/.l3m_chat_history`
- Consolidate all l3m data under `~/.l3m/` directory
- Maintain backwards compatibility or migrate existing history

---

## Note 13 — DONE

**Raw:**
> Handle mcp exceptions in l3m-chat and dump them into a ~/.l3m/logs/<session-filename>.log file.

**Revised:**
Add MCP exception logging to l3m-chat:
- Catch all MCP-related exceptions (connection errors, tool errors, etc.)
- Write full tracebacks to `~/.l3m/logs/<session-filename>.log`
- Keep user-facing output clean (no tracebacks in terminal)
- Preserve debug info in log files for troubleshooting

---

## Note 14 — DONE

**Raw:**
> Add --summary-ctx and --transcript-ctx to the config file.

**Revised:**
Add `summary_ctx` and `transcript_ctx` settings to the l3m config file (`~/.l3m/config.yaml` or similar):
- Allow users to set default values for `--summary-ctx` and `--transcript-ctx` flags
- CLI flags should override config file values
- Avoids needing to specify these flags on every invocation

---

## Note 15 — DONE

**Raw:**
> Pure and magic commands inside ~/.l3m/commands and ~/.l3m/magic should be symlinks to the builtin pure and magic commands.

**Revised:**
Populate `~/.l3m/commands/` and `~/.l3m/magic/` with symlinks to built-in commands:
- On first run or setup, create symlinks pointing to the package's built-in command implementations
- Users can remove symlinks to disable built-ins, or replace them with custom implementations
- Provides visibility into available commands and easy customization
- Keeps built-in source in the package while allowing user overrides

---

## Note 16 — DONE

**Raw:**
> Builtin tools should be symlinked into ~/.l3m/tools/

**Revised:**
Populate `~/.l3m/tools/` with symlinks to built-in tools:
- On first run or setup, create symlinks pointing to the package's built-in tool implementations
- Users can remove symlinks to disable built-in tools, or replace them with custom implementations
- Consistent with the symlink approach for commands and magic commands (Notes 15, 10, 11)
- Provides a unified plugin architecture across tools, commands, and magic commands

---

## Note 17 — DONE

**Raw:**
> Add a prefix `#` for MCP prompt expansion and their arguments with autocompletion. The idea is that the use invokes the mcp prompt template and the MCP server return a revised prompt.

**Revised:**
Add `#` prefix for MCP prompt template expansion:
- `#<prompt_name>` invokes an MCP prompt template
- Autocompletion for available prompt names from connected MCP servers
- Autocompletion for prompt arguments (e.g., `#summarize text="..."`)
- When invoked, the MCP server returns an expanded/revised prompt
- The expanded prompt is then used as the user's input to the model
- Similar to `@` for resources, `#` is for prompts

---

## Note 18

**Raw:**
> Add remote MCP error handling to the todo list.

**Revised:**
Improve remote MCP error handling throughout the stack:
- Retry logic with exponential backoff for transient failures (DONE in transport.py)
- Wrap low-level errors (httpx, connection) in MCPTransportError (DONE)
- Log full tracebacks to `~/.l3m/logs/` for debugging (DONE - see Note 13)
- Graceful degradation when MCP servers are unavailable
- User-friendly error messages in the REPL (no raw tracebacks)
- Handle server disconnects during tool execution with clear feedback

---

## Note 19

**Raw:**
> Use similarity graph to trim history when needed. Work from inside-out (e.g. choose the middle history entry, compare it to entries before it and after it, if it has high similarity trim it). What other policies can we adopt when trimming the context.

**Revised:**
**Context Trimming via Similarity Graph**

Use the similarity graph to intelligently trim conversation history when context limits are reached.

**Inside-Out Policy (proposed):**
- Start from the middle of history
- Compare each entry to its neighbors (before/after)
- If high similarity to neighbors, trim it (redundant information)
- Preserves beginning (context setup) and end (recent exchanges)

**Other potential trimming policies:**
- **Recency-weighted:** Preserve recent messages, aggressively trim old ones
- **Role-based:** Keep all user messages, trim verbose assistant responses
- **Semantic clustering:** Keep one representative from each topic cluster
- **Importance scoring:** Preserve messages with tool calls, code, or key decisions
- **Sliding window + anchors:** Keep first N + last M, trim middle intelligently
- **Summary replacement:** Replace trimmed sections with generated summaries
- **Low-information removal:** Trim "okay", "thanks", acknowledgments first
- **Duplicate elimination:** Remove near-duplicate messages (similarity > 0.9)

---
