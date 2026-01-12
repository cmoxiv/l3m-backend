"""
Chat Engine for LLM interaction with tool calling.
"""

from __future__ import annotations

import json
import sys
import time
from typing import TYPE_CHECKING, Any, Callable

from l3m_backend.core import ToolOutput, ToolRegistry
from l3m_backend.engine.context import ContextPartition, set_current_engine, reset_current_engine
from l3m_backend.engine.contract import MINIMAL_CONTRACT_TEMPLATE, load_contract_template
from l3m_backend.engine.legacy_priming import generate_legacy_priming
from l3m_backend.engine.toolcall import ToolCallFormat, get_handler

if TYPE_CHECKING:
    from llama_cpp import Llama
    from l3m_backend.engine.knowledge import KnowledgeGraph
    from l3m_backend.engine.similarity import SimilarityGraph


class ChatEngine:
    """Chat engine that manages LLM interaction with contract-based tool calling."""

    def __init__(
        self,
        model_path: str,
        registry: ToolRegistry,
        system_prompt: str | None = None,
        n_ctx: int = 32768,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        show_warnings: bool = False,
        use_native_tools: bool = True,
        chat_format: str | None = None,
        temperature: float = 0.0,
        flash_attn: bool = True,
        debug: bool = False,
        minimal_contract: bool = False,
        context_partition: ContextPartition | None = None,
        stop_tokens: list[str] | None = None,
    ):
        # Import here to make llama-cpp-python optional
        from llama_cpp import Llama

        self.registry = registry
        self.use_native_tools = use_native_tools
        self.temperature = temperature
        self.debug = debug
        self.minimal_contract = minimal_contract
        self._n_gpu_layers = n_gpu_layers  # Store for GPU detection
        self.stop_tokens = stop_tokens or []  # Stop tokens for generation

        # Suppress stderr during model loading (hides Metal "skipping kernel" messages)
        if not show_warnings:
            import os
            stderr_fd = sys.stderr.fileno()
            old_stderr = os.dup(stderr_fd)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, stderr_fd)
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    verbose=verbose,
                    chat_format=chat_format,
                    flash_attn=flash_attn,
                )
            finally:
                os.dup2(old_stderr, stderr_fd)
                os.close(old_stderr)
                os.close(devnull)
        else:
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=verbose,
                chat_format=chat_format,
                flash_attn=flash_attn,
            )
        self.system_prompt = system_prompt or "You are a helpful assistant with access to tools."
        self.tools = registry.to_openai_tools()

        # Load priming messages (mock conversation examples)
        self.priming_messages = generate_legacy_priming(registry)

        # Build tools contract - minimal or full
        if minimal_contract:
            self.tools_contract = MINIMAL_CONTRACT_TEMPLATE.format(
                tool_list=registry.to_minimal_list(),
            )
        else:
            self.tools_contract = load_contract_template().format(
                registry_json=registry.to_registry_json(),
            )

        if self.debug:
            print(f"\033[90m[DEBUG] Tools contract: {len(self.tools_contract)} chars\033[0m", file=sys.stderr)

        self.history: list[dict[str, Any]] = []
        self.history_trimmed: bool = False  # Flag to signal when history was trimmed
        self.partition = context_partition or ContextPartition()

        # Similarity graph (built during warmup if requested)
        self.similarity_graph: "SimilarityGraph | None" = None
        self._embedding_llm: "Llama | None" = None  # Cached embedding model
        self._embedding_provider: Any = None  # Cached embedding provider (Nomic or Llama)
        self._model_path = model_path  # Store for embedding model creation
        self._use_nomic_embeddings = True  # Prefer Nomic if available
        self._embeddings_enabled = False  # Disabled by default, enable with --embedding-model

        # Knowledge graph (built during warmup if requested)
        self.knowledge_graph: "KnowledgeGraph | None" = None

    def _debug_print(self, msg: str) -> None:
        """Print debug message to stderr if debug mode is enabled."""
        if self.debug:
            print(f"\033[90m[DEBUG] {msg}\033[0m", file=sys.stderr)

    def _build_system_message(self) -> dict[str, str]:
        """Build system message with tools contract.

        Combines the base system prompt with the tool calling contract,
        which includes tool specifications and output format instructions.

        Returns:
            System message dict with role and content.
        """
        content = f"{self.system_prompt}\n\n{self.tools_contract}"
        return {"role": "system", "content": content}

    def _build_messages(self, pending: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Construct full message list for LLM.

        Builds the complete message context by combining:
        1. System message (with tool contract)
        2. Priming messages (mock conversation examples)
        3. Partitioned context (summaries and/or transcript excerpts)
        4. Conversation history
        5. Any pending messages (current turn)

        Args:
            pending: Optional list of messages for the current turn
                    (e.g., user input and tool results not yet in history).

        Returns:
            Complete list of messages to send to the LLM.
        """
        messages = [self._build_system_message()]

        # Add priming messages (mock conversation examples)
        messages.extend(self.priming_messages)

        # Add partitioned context if available
        if self.partition.has_partitions():
            # Add loaded summaries
            if self.partition.loaded_summaries:
                summary_content = "\n\n".join(self.partition.loaded_summaries)
                messages.append({
                    "role": "system",
                    "content": f"Previous session context:\n{summary_content}",
                })

            # Add loaded transcript excerpts
            if self.partition.loaded_transcript:
                transcript_lines = []
                for msg in self.partition.loaded_transcript:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if content:
                        transcript_lines.append(f"{role}: {content[:300]}")
                if transcript_lines:
                    transcript_content = "\n".join(transcript_lines)
                    messages.append({
                        "role": "system",
                        "content": f"Previous transcript:\n{transcript_content}",
                    })

        messages.extend(self.history)
        if pending:
            messages.extend(pending)
        return messages

    def _result2llm(self, output: Any) -> str:
        """Convert tool output to LLM-friendly string.

        Args:
            output: Raw tool output (ToolOutput or other).

        Returns:
            String representation using llm_format if available,
            otherwise JSON-serialized output.
        """
        if isinstance(output, ToolOutput) and output.llm_format:
            return output.llm_format
        elif isinstance(output, ToolOutput):
            return json.dumps(output.data)
        else:
            return json.dumps(output)

    def _execute_tool_by_name(self, name: str, args: dict) -> dict[str, Any]:
        """Execute a tool by name and return the result.

        Executes the tool through the registry, which handles validation
        and execution. Validates that required arguments are not empty.
        Sets the current engine context so tools can access it.

        Args:
            name: Tool name or alias to execute.
            args: Dictionary of arguments to pass to the tool.

        Returns:
            Dictionary with name, arguments, and output.
            Errors are returned with an "error" key in output.
        """
        # Set engine context so tools can access it
        token = set_current_engine(self)
        try:
            # Validate required arguments are not empty
            entry = self.registry.get(name)
            schema = entry.get_params_model().model_json_schema()
            required = schema.get("required", [])

            empty_required = [
                k for k in required
                if k in args and args[k] == ""
            ]
            if empty_required:
                missing = ", ".join(empty_required)
                return {
                    "name": name,
                    "arguments": args,
                    "output": {"error": f"Required argument(s) cannot be empty: {missing}. Ask the user for the value."},
                }

            result = self.registry.execute(name, args)
            return {
                "name": result.name,
                "arguments": result.arguments,
                "output": result.output,
            }
        except Exception as e:
            return {
                "name": name,
                "arguments": args,
                "output": {"error": str(e)},
            }
        finally:
            reset_current_engine(token)

    def _execute_tool(self, tool_call: dict) -> dict[str, Any]:
        """Execute an OpenAI-format tool call.

        Parses OpenAI-style tool call format and delegates to _execute_tool_by_name.
        Currently unused but available for compatibility with OpenAI tool call format.

        Args:
            tool_call: OpenAI-format tool call dict with structure:
                      {"function": {"name": "...", "arguments": {...}}}

        Returns:
            Dictionary with name, arguments, and output.
        """
        name = tool_call["function"]["name"]
        args_json = tool_call["function"]["arguments"]
        args = json.loads(args_json) if isinstance(args_json, str) else args_json
        return self._execute_tool_by_name(name, args)

    def _parse_tool_call(self, content: str | None) -> dict | None:
        """Parse tool call from LLM response content.

        Uses the ToolCallHandler to parse tool calls from various formats
        including contract, OpenAI, Hermes, and raw JSON formats.

        Args:
            content: Raw content string from LLM response.

        Returns:
            Normalized dict {"type": "tool_call", "name": "...", "arguments": {...}}
            if valid tool call found, otherwise None.
        """
        if not content:
            return None

        handler = get_handler()
        tool_call = handler.parse(content)
        if tool_call:
            return tool_call.to_dict()
        return None

    def _extract_embedded_tool_call(self, content: str) -> dict | None:
        """Extract a tool call JSON embedded in text response.

        Handles cases where weaker models output mixed text and JSON:
        "Let me help... {"type": "tool_call", "name": "...", "arguments": {...}}"

        Args:
            content: Full response content that may contain embedded JSON.

        Returns:
            Extracted tool call dict, or None if not found.
        """
        if not content:
            return None

        import re
        # Find JSON object containing "type": "tool_call"
        # This pattern handles nested braces in arguments
        pattern = r'\{\s*"type"\s*:\s*"tool_call"[^}]*"arguments"\s*:\s*\{[^}]*\}[^}]*\}'
        match = re.search(pattern, content)
        if match:
            try:
                parsed = json.loads(match.group())
                if parsed.get("type") == "tool_call" and parsed.get("name"):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: simpler pattern for flat arguments
        pattern_simple = r'\{\s*"type"\s*:\s*"tool_call"\s*,\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}'
        match = re.search(pattern_simple, content)
        if match:
            try:
                parsed = json.loads(match.group())
                if parsed.get("type") == "tool_call" and parsed.get("name"):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    def _chat_generator(self, user_input: str, max_tool_rounds: int = 3):
        """Internal generator that yields tokens during streaming.

        Yields:
            str: Empty string as sentinel to stop animation, then tokens.
        """
        self._trim_history()
        user_msg = {"role": "user", "content": user_input}
        pending: list[dict[str, Any]] = [user_msg]
        called_tools: set[str] = set()

        if self.debug:
            print(file=sys.stderr)  # Newline before debug output

        for round_num in range(max_tool_rounds):
            self._debug_print(f"Round {round_num + 1}/{max_tool_rounds}")

            t_start = time.perf_counter()
            messages = self._build_messages(pending)
            t_build = time.perf_counter()
            total_chars = sum(len(m.get("content", "")) for m in messages)
            self._debug_print(f"  build_messages: {(t_build - t_start)*1000:.2f}ms ({len(messages)} msgs, {total_chars} chars)")

            self.llm.reset()
            t_reset = time.perf_counter()
            self._debug_print(f"  llm.reset: {(t_reset - t_build)*1000:.2f}ms")

            # Use streaming to collect response
            if self.use_native_tools:
                llm_stream = self.llm.create_chat_completion(
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    stream=True,
                    stop=self.stop_tokens if self.stop_tokens else None,
                )
            else:
                llm_stream = self.llm.create_chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                    stop=self.stop_tokens if self.stop_tokens else None,
                )

            # Stream tokens with mode detection:
            # - Native tool calls: collect all (detected via delta["tool_calls"])
            # - JSON contract tool calls: collect all (detected if starts with '{')
            # - Final response: stream immediately
            content = ""
            tool_calls_data: list[dict[str, Any]] = []
            t_stream_start = time.perf_counter()
            streaming_mode = False  # True = streaming final response
            first_content_seen = False

            for chunk in llm_stream:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                # Check for native tool calls - always collect mode
                if self.use_native_tools and "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        while len(tool_calls_data) <= idx:
                            tool_calls_data.append({"id": "", "function": {"name": "", "arguments": ""}})
                        if "id" in tc:
                            tool_calls_data[idx]["id"] = tc["id"]
                        if "function" in tc:
                            if "name" in tc["function"]:
                                tool_calls_data[idx]["function"]["name"] = tc["function"]["name"]
                            if "arguments" in tc["function"]:
                                tool_calls_data[idx]["function"]["arguments"] += tc["function"]["arguments"]
                    continue

                token = delta.get("content", "")
                if token:
                    content += token

                    # Decide mode on first non-whitespace content
                    if not first_content_seen:
                        # Skip leading whitespace when deciding mode
                        stripped = content.lstrip()
                        if stripped:
                            first_content_seen = True
                            # If starts with '{', likely JSON tool call - collect mode
                            if stripped.startswith("{"):
                                streaming_mode = False
                            else:
                                # Final response - stream mode
                                streaming_mode = True
                                self._debug_print(f"  streaming: enabled")
                                yield ""  # Sentinel to stop animation
                                # Yield accumulated content (without leading whitespace)
                                yield stripped

                    # Stream token if in streaming mode (skip if we just yielded accumulated)
                    elif streaming_mode:
                        yield token

            t_stream_end = time.perf_counter()
            self._debug_print(f"  llm.generate: {(t_stream_end - t_stream_start)*1000:.2f}ms ({len(content)} chars)")

            # Check if it's a native tool call
            if tool_calls_data:
                self._debug_print(f"  detected: native tool call")
                assistant_msg = {"role": "assistant", "content": content, "tool_calls": tool_calls_data}
                pending.append(assistant_msg)
                has_new_calls = False

                for tool_call in tool_calls_data:
                    name = tool_call["function"]["name"]
                    args_json = tool_call["function"]["arguments"]

                    t_parse_args = time.perf_counter()
                    args = json.loads(args_json) if isinstance(args_json, str) else args_json
                    self._debug_print(f"  parse_args ({name}): {(time.perf_counter() - t_parse_args)*1000:.2f}ms")

                    call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                    if call_key in called_tools:
                        pending.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": "[Already called - use the previous result to respond]",
                        })
                        continue
                    called_tools.add(call_key)
                    has_new_calls = True

                    t_exec = time.perf_counter()
                    result = self._execute_tool_by_name(name, args)
                    self._debug_print(f"  execute_tool ({name}): {(time.perf_counter() - t_exec)*1000:.2f}ms")

                    t_format = time.perf_counter()
                    pending.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": self._result2llm(result["output"]),
                    })
                    self._debug_print(f"  format_result: {(time.perf_counter() - t_format)*1000:.2f}ms")

                if not has_new_calls:
                    self.history.extend(pending)
                    return
                continue

            # If we were in streaming mode, we've already yielded everything
            if streaming_mode:
                self._debug_print(f"  detected: final response (streamed)")

                # Check for embedded tool call in streamed response (weaker models)
                embedded_tool = self._extract_embedded_tool_call(content)
                if embedded_tool:
                    self._debug_print(f"  detected: embedded tool call ({embedded_tool.get('name')})")
                    # Yield special marker for REPL to handle user confirmation
                    yield {"type": "tool_suggestion", "tool": embedded_tool, "preamble": content}

                pending.append({"role": "assistant", "content": content})
                self.history.extend(pending)
                self._debug_print(f"  total round: {(time.perf_counter() - t_start)*1000:.2f}ms")
                return

            # Not streaming - check if it's a JSON contract tool call
            t_parse = time.perf_counter()
            parsed = self._parse_tool_call(content)
            self._debug_print(f"  parse_content_json: {(time.perf_counter() - t_parse)*1000:.2f}ms")

            if parsed and parsed["type"] == "tool_call":
                self._debug_print(f"  detected: contract tool call ({parsed.get('name')})")
                pending.append({"role": "assistant", "content": content})
                name = parsed.get("name", "")
                args = parsed.get("arguments", {})

                call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                if call_key in called_tools:
                    self.history.extend(pending)
                    return
                called_tools.add(call_key)

                t_exec = time.perf_counter()
                result = self._execute_tool_by_name(name, args)
                self._debug_print(f"  execute_tool ({name}): {(time.perf_counter() - t_exec)*1000:.2f}ms")

                t_format = time.perf_counter()
                output = self._result2llm(result["output"])

                pending.append({
                    "role": "tool",
                    "content": json.dumps({"type": "tool_result", "name": name, "content": output}),
                })
                self._debug_print(f"  format_result: {(time.perf_counter() - t_format)*1000:.2f}ms")
                continue

            else:
                # Collected but not a tool call - yield as final response
                self._debug_print(f"  detected: final response (collected)")
                pending.append({"role": "assistant", "content": content})
                self.history.extend(pending)
                self._debug_print(f"  total round: {(time.perf_counter() - t_start)*1000:.2f}ms")
                yield ""  # Sentinel
                yield content  # Full content
                return

        # Max rounds reached
        self.history.extend(pending)
        yield "[Max tool rounds reached]"

    def chat(
        self,
        user_input: str,
        max_tool_rounds: int = 3,
        stream: bool = False,
        ignore_history: bool = False,
        temp_system: str | None = None,
    ):
        """Send a message and get a response with automatic tool execution.

        Implements a multi-turn conversation loop that handles tool calls
        transparently. The LLM can call tools by responding with JSON
        in the format: {"type": "tool_call", "name": "...", "arguments": {...}}

        The loop continues until:
        1. The LLM provides a plain text response (final)
        2. Max rounds reached

        Args:
            user_input: The user's message.
            max_tool_rounds: Maximum number of LLM generation rounds to allow.
                           Prevents infinite loops. Default is 3.
            stream: If True, returns a generator yielding tokens.
                   If False, returns the complete response string.
            ignore_history: If True, bypass history and tools for a one-shot
                          completion. Useful for tools that need LLM processing.
            temp_system: Temporary system message (only used with ignore_history).

        Returns:
            str or Iterator[str]: The response (string if stream=False,
                                 generator if stream=True).

        Note:
            - Automatically resets model state before each generation (required
              for hybrid/Mamba models)
            - All messages (including tool results) are committed to history
            - History is trimmed if it exceeds context limit
        """
        # Handle ignore_history mode - simple one-shot completion
        if ignore_history:
            if stream:
                raise ValueError("stream must be False when ignore_history=True")
            return self._chat_ignore_history(user_input, temp_system)

        if stream:
            return self._chat_generator(user_input, max_tool_rounds)

        # Non-streaming path
        return self._chat_non_streaming(user_input, max_tool_rounds)

    def _chat_ignore_history(self, user_input: str, temp_system: str | None = None) -> str:
        """One-shot chat completion without history or tool calling.

        Used by tools that need LLM processing (e.g., summarization)
        without affecting the main conversation.

        Args:
            user_input: The text to process.
            temp_system: Optional temporary system message.

        Returns:
            The LLM response as a string.
        """
        messages = []

        # Use temp_system or default
        system_content = temp_system or "You are a helpful assistant."
        messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_input})

        self.llm.reset()
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            stop=self.stop_tokens if self.stop_tokens else None,
        )

        return response["choices"][0]["message"].get("content", "")

    def _chat_non_streaming(self, user_input: str, max_tool_rounds: int = 3) -> str:
        """Non-streaming chat implementation."""
        self._trim_history()

        user_msg = {"role": "user", "content": user_input}
        pending: list[dict[str, Any]] = [user_msg]
        called_tools: set[str] = set()

        for round_num in range(max_tool_rounds):
            messages = self._build_messages(pending)
            self.llm.reset()

            # Call LLM with or without native tools
            if self.use_native_tools:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    stop=self.stop_tokens if self.stop_tokens else None,
                )
            else:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    stop=self.stop_tokens if self.stop_tokens else None,
                )

            assistant_msg = response["choices"][0]["message"]
            content = assistant_msg.get("content", "")

            # Check for native tool calls first (if enabled)
            if self.use_native_tools and assistant_msg.get("tool_calls"):
                pending.append(assistant_msg)
                has_new_calls = False
                for tool_call in assistant_msg["tool_calls"]:
                    name = tool_call["function"]["name"]
                    args_json = tool_call["function"]["arguments"]
                    args = json.loads(args_json) if isinstance(args_json, str) else args_json

                    call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                    if call_key in called_tools:
                        pending.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": "[Already called - use the previous result to respond]",
                        })
                        continue
                    called_tools.add(call_key)
                    has_new_calls = True

                    result = self._execute_tool_by_name(name, args)
                    pending.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": self._result2llm(result["output"]),
                    })
                if not has_new_calls:
                    self.history.extend(pending)
                    return "[Tool already executed - please provide a response]"
                continue

            # Try to parse as our JSON contract format (fallback)
            parsed = self._parse_tool_call(content)

            if parsed and parsed["type"] == "tool_call":
                pending.append(assistant_msg)
                name = parsed.get("name", "")
                args = parsed.get("arguments", {})

                call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                if call_key in called_tools:
                    self.history.extend(pending)
                    return "[Tool already executed - please provide a response]"
                called_tools.add(call_key)

                result = self._execute_tool_by_name(name, args)
                output = self._result2llm(result["output"])

                
                pending.append({
                    "role": "tool",
                    "content": json.dumps({"type": "tool_result", "name": name, "content": output}),
                })
                continue

            else:
                pending.append(assistant_msg)
                self.history.extend(pending)
                return content

        self.history.extend(pending)
        return "[Max tool rounds reached]"

    def clear(self):
        """Clear conversation history.

        Resets the conversation to a clean slate. System message
        (with tool contract) is NOT affected - only history and loaded
        partition context (summaries/transcript) are cleared.
        """
        self.history = []
        # Also clear loaded partition context
        self.partition.loaded_summaries = []
        self.partition.loaded_transcript = []

    def execute_suggested_tool(self, tool_call: dict) -> str:
        """Execute a tool that was suggested (embedded in streamed response).

        Called by REPL after user confirms they want to execute
        the tool that was detected in a mixed text+JSON response.

        Args:
            tool_call: Dict with "name" and "arguments" keys.

        Returns:
            The tool result formatted for display.
        """
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        result = self._execute_tool_by_name(name, args)
        output = self._result2llm(result["output"])

        # Add tool result to history
        self.history.append({
            "role": "tool",
            "content": json.dumps({"type": "tool_result", "name": name, "content": output}),
        })

        return output

    def _count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens for a list of messages.

        Uses the model's tokenizer for accurate counting.

        Args:
            messages: List of message dicts with role and content.

        Returns:
            Total token count for all messages.
        """
        text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            text += f"{role}: {content}\n"

        try:
            tokens = self.llm.tokenize(text.encode())
            return len(tokens)
        except Exception:
            # Fallback: estimate 4 chars per token
            return len(text) // 4

    def _trim_history(self, reserve_tokens: int = 2048) -> None:
        """Trim history to fit within context limit.

        Drops oldest history messages until the remaining context fits.
        System message and pending messages are NEVER touched.
        Sets history_trimmed flag if any messages were removed.

        Args:
            reserve_tokens: Space to reserve for response generation.
        """
        max_tokens = self.llm.n_ctx() - reserve_tokens

        # Calculate fixed cost (system + contract)
        system_tokens = self._count_tokens([self._build_system_message()])

        available = max_tokens - system_tokens
        if available <= 0:
            # System alone exceeds limit - nothing we can do
            return

        # Drop oldest messages until history fits
        trimmed = False
        while self.history:
            history_tokens = self._count_tokens(self.history)
            if history_tokens <= available:
                break
            # Remove oldest message
            self.history.pop(0)
            trimmed = True

        if trimmed:
            self.history_trimmed = True

    def check_and_trim_history(self, reserve_tokens: int = 2048) -> dict[str, Any]:
        """Check if history exceeds context and trim if needed.

        Call this after loading a session to ensure history fits the current
        context size (which may differ from when the session was saved).

        Args:
            reserve_tokens: Space to reserve for response generation.

        Returns:
            Dict with 'trimmed' (bool), 'removed_count' (int), 'remaining_count' (int),
            'context_size' (int), 'history_tokens' (int).
        """
        original_count = len(self.history)
        max_tokens = self.llm.n_ctx()

        # Get initial token count
        initial_tokens = self._count_tokens(self.history) if self.history else 0

        # Trim if needed
        self._trim_history(reserve_tokens)

        removed_count = original_count - len(self.history)
        final_tokens = self._count_tokens(self.history) if self.history else 0

        return {
            "trimmed": removed_count > 0,
            "removed_count": removed_count,
            "remaining_count": len(self.history),
            "context_size": max_tokens,
            "initial_tokens": initial_tokens,
            "final_tokens": final_tokens,
        }

    # For REPL compatibility
    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return full message list for display."""
        return self._build_messages()

    def switch_model(
        self,
        model_path: str,
        preserve_history: bool = True,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
    ) -> None:
        """Switch to a different model, optionally preserving history.

        Args:
            model_path: Path to the new model file.
            preserve_history: Keep conversation history (default True).
            n_ctx: Context size (default: keep current).
            n_gpu_layers: GPU layers (default: keep current).
        """
        from llama_cpp import Llama

        # Save history if requested
        old_history = self.history.copy() if preserve_history else []

        # Get current settings for defaults
        current_n_ctx = self.llm.n_ctx()
        current_n_gpu_layers = self._n_gpu_layers

        # Cleanup old model
        del self.llm

        # Load new model
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers if n_gpu_layers is not None else current_n_gpu_layers,
            n_ctx=n_ctx if n_ctx is not None else current_n_ctx,
            verbose=False,
            flash_attn=True,
        )

        # Update stored settings
        if n_gpu_layers is not None:
            self._n_gpu_layers = n_gpu_layers

        # Restore history
        self.history = old_history

        # Reset trimmed flag
        self.history_trimmed = False

    def warmup(
        self,
        transcript: list[dict[str, Any]] | None = None,
        summaries: list[str] | None = None,
        max_transcript_messages: int = 20,
        build_similarity_graph: bool = False,
        build_knowledge_graph: bool = False,
        kg_progress_callback: Callable[[int, int], None] | None = None,
        sg_progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Run warmup generation to prime KV cache.

        Processes historical context and requests a short summary,
        but discards all outputs (no state changes to history).

        This is useful when loading a session to reduce latency
        on the first actual user interaction.

        Args:
            transcript: Optional list of message dicts to include in warmup.
            summaries: Optional list of summary strings to include.
            max_transcript_messages: Max transcript messages to use (default 20).
            build_similarity_graph: If True, build similarity graph from transcript/summaries.
            build_knowledge_graph: If True, build knowledge graph from transcript.
            kg_progress_callback: Optional callback(current, total) for knowledge graph progress.
            sg_progress_callback: Optional callback(current, total) for similarity graph progress.

        Returns:
            Dict with warmup info: {"tokens": N, "time_s": X, "graph_built": bool, "kg_built": bool}
        """
        # Build warmup messages
        messages = [self._build_system_message()]

        # Add summaries if provided
        if summaries:
            for summary in summaries:
                messages.append({"role": "system", "content": f"Previous context: {summary}"})

        # Add transcript messages (limited)
        if transcript:
            for msg in transcript[-max_transcript_messages:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        # Add warmup prompt (request a short completion)
        messages.append({
            "role": "user",
            "content": "Summarize the conversation so far in one sentence.",
        })

        # Run generation (output is discarded)
        t_start = time.perf_counter()
        self.llm.reset()
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=50,  # Short response
            temperature=0.1,
            stop=self.stop_tokens if self.stop_tokens else None,
        )
        elapsed = time.perf_counter() - t_start

        # Extract token count
        usage = response.get("usage", {})
        total_tokens = usage.get("total_tokens", 0) or usage.get("prompt_tokens", 0)

        # Build similarity graph if requested
        graph_built = False
        if build_similarity_graph and (transcript or summaries):
            try:
                self._build_similarity_graph(
                    transcript or [],
                    summaries or [],
                    progress_callback=sg_progress_callback,
                )
                graph_built = self.similarity_graph is not None and self.similarity_graph.is_built
            except Exception as e:
                print(f"\033[33mFailed to build similarity graph: {e}\033[0m", file=sys.stderr)

        # Build knowledge graph if requested
        kg_built = False
        if build_knowledge_graph and transcript:
            try:
                self._build_knowledge_graph(transcript, progress_callback=kg_progress_callback)
                kg_built = self.knowledge_graph is not None
            except Exception as e:
                self._debug_print(f"Failed to build knowledge graph: {e}")

        # DO NOT add to history - this is warmup only
        return {
            "tokens": total_tokens,
            "time_s": elapsed,
            "graph_built": graph_built,
            "kg_built": kg_built,
        }

    def _build_similarity_graph(
        self,
        transcript: list[dict[str, Any]],
        summaries: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Build similarity graph from transcript and summaries.

        Uses Nomic embeddings (preferred) or LLM-based embeddings.

        Args:
            transcript: List of message dicts with 'role' and 'content'
            summaries: List of summary strings
            progress_callback: Optional callback(current, total) for progress updates
        """
        from l3m_backend.engine.similarity import SimilarityGraph

        # Get embedding provider (Nomic preferred, falls back to LLM)
        provider = self._get_embedding_provider()
        if provider is None:
            print("\033[33mNo embedding provider available for similarity graph\033[0m", file=sys.stderr)
            print("\033[33mInstall with: pip install sentence-transformers\033[0m", file=sys.stderr)
            return

        # Build graph
        self.similarity_graph = SimilarityGraph()
        self.similarity_graph.build(transcript, summaries, provider, progress_callback=progress_callback)

        if self.debug:
            stats = self.similarity_graph.stats
            self._debug_print(f"Similarity graph: {stats}")

    def _get_embedding_provider(self) -> Any:
        """Get or create an embedding provider.

        Prefers Nomic embeddings if available (faster, better quality).
        Falls back to LLM-based embeddings.

        Returns:
            EmbeddingProvider instance, or None if no provider available
        """
        # Return cached provider if available
        if self._embedding_provider is not None:
            return self._embedding_provider

        # Check if embeddings are enabled (set via _use_nomic_embeddings being explicitly set)
        # If _use_nomic_embeddings is still default True but not explicitly enabled, skip
        if not hasattr(self, '_embeddings_enabled') or not self._embeddings_enabled:
            return None

        from l3m_backend.engine.similarity import get_embedding_provider

        # Try to get provider (prefers Nomic, falls back to LLM)
        # Only create LLM embedding fallback if Nomic fails (avoid loading model twice on GPU)
        if self._use_nomic_embeddings:
            # Try Nomic first
            self._embedding_provider = get_embedding_provider(
                llm=None,
                use_nomic=True,
            )
            # If Nomic failed, try LLM fallback
            if self._embedding_provider is None:
                embedding_llm = self._get_embedding_llm()
                if embedding_llm:
                    from l3m_backend.engine.similarity import LlamaEmbeddingProvider
                    self._embedding_provider = LlamaEmbeddingProvider(embedding_llm)
        else:
            # User explicitly requested LLM embeddings
            embedding_llm = self._get_embedding_llm()
            self._embedding_provider = get_embedding_provider(
                llm=embedding_llm,
                use_nomic=False,
            )

        if self._embedding_provider is not None:
            provider_type = type(self._embedding_provider).__name__
            self._debug_print(f"Using embedding provider: {provider_type}")

        return self._embedding_provider

    def _get_embedding_llm(self) -> "Llama | None":
        """Get or create an embedding-capable Llama instance.

        Uses the same context size as the main model.
        No system prompt or contract - pure embeddings only.

        Returns:
            Llama instance with embedding support, or None if failed
        """
        # Return cached embedding LLM if available
        if self._embedding_llm is not None:
            return self._embedding_llm

        # Create a new Llama instance with embedding=True
        try:
            from llama_cpp import Llama
            import os

            # Use small context for embeddings (large contexts cause huge buffer allocations)
            # Embeddings only need enough context for single text chunks, not full conversations
            n_ctx = 512

            # Suppress stderr during model loading
            stderr_fd = sys.stderr.fileno()
            old_stderr = os.dup(stderr_fd)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, stderr_fd)
            try:
                self._embedding_llm = Llama(
                    model_path=self._model_path,
                    n_gpu_layers=self._n_gpu_layers,
                    n_ctx=n_ctx,
                    embedding=True,
                    verbose=False,
                )
                # Set stop tokens (used if model generates text during pooling)
                self._embedding_llm.set_seed(42)  # Deterministic embeddings
            finally:
                os.dup2(old_stderr, stderr_fd)
                os.close(old_stderr)
                os.close(devnull)

            return self._embedding_llm
        except Exception as e:
            print(f"\033[33mFailed to create embedding LLM: {e}\033[0m", file=sys.stderr)
            return None

    def _build_knowledge_graph(
        self,
        transcript: list[dict[str, Any]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Build knowledge graph from transcript.

        Uses LLM-based semantic entity extraction. Computes contextual embeddings
        using Nomic (preferred) or LLM embeddings.

        Args:
            transcript: List of message dicts with 'role' and 'content'
            progress_callback: Optional callback(current, total) for progress
        """
        from l3m_backend.engine.extraction import build_knowledge_graph_from_transcript

        # Get embedding provider (Nomic preferred, falls back to LLM)
        embedding_provider = self._get_embedding_provider()

        # Build graph with LLM-based semantic extraction and embeddings
        self.knowledge_graph = build_knowledge_graph_from_transcript(
            transcript,
            llm=self.llm,  # Enable semantic entity extraction
            progress_callback=progress_callback,
            embedding_provider=embedding_provider,
        )

        if self.debug:
            stats = self.knowledge_graph.stats
            has_emb = self.knowledge_graph.has_embeddings
            self._debug_print(f"Knowledge graph: {stats}, embeddings: {has_emb}")

    @property
    def has_knowledge_graph(self) -> bool:
        """Check if knowledge graph is available."""
        return self.knowledge_graph is not None and len(self.knowledge_graph.entities) > 0

    @property
    def has_similarity_graph(self) -> bool:
        """Check if similarity graph is available."""
        return self.similarity_graph is not None and self.similarity_graph.is_built

    def get_relevant_context_for_query(
        self,
        query: str,
        max_messages: int = 10,
        max_summaries: int = 3,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Get relevant historical messages and summaries for a query.

        Uses similarity graph if available, otherwise returns recent messages.

        Args:
            query: User's current query
            max_messages: Maximum messages to return
            max_summaries: Maximum summaries to return

        Returns:
            Tuple of (relevant_messages, relevant_summaries)
        """
        if not self.has_similarity_graph:
            # Fallback: return last N messages from history
            return self.history[-max_messages:], []

        # Get query embedding
        embedding_llm = self._get_embedding_llm()
        if embedding_llm is None:
            return self.history[-max_messages:], []

        from l3m_backend.engine.similarity import LlamaEmbeddingProvider
        provider = LlamaEmbeddingProvider(embedding_llm)
        query_embedding = provider.embed_single(query)

        # Get relevant indices from graph
        msg_indices, sum_indices = self.similarity_graph.get_relevant_context(
            query_embedding,
            max_messages=max_messages,
            max_summaries=max_summaries,
        )

        # Collect messages (from the transcript stored in graph nodes)
        relevant_messages = []
        for idx in msg_indices:
            if idx < len(self.similarity_graph.message_nodes):
                node = self.similarity_graph.message_nodes[idx]
                relevant_messages.append({
                    "role": "user",  # Simplified - actual role lost in graph
                    "content": node.content,
                })

        # Collect summaries
        relevant_summaries = []
        for idx in sum_indices:
            if idx < len(self.similarity_graph.summary_nodes):
                node = self.similarity_graph.summary_nodes[idx]
                relevant_summaries.append(node.content)

        return relevant_messages, relevant_summaries
