"""
Chat Engine for LLM interaction with tool calling.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from l3m_backend.core import ToolOutput, ToolRegistry
from l3m_backend.engine.context import ContextPartition
from l3m_backend.engine.contract import MINIMAL_CONTRACT_TEMPLATE, TOOL_CONTRACT_TEMPLATE
from l3m_backend.engine.toolcall import ToolCallFormat, get_handler


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
    ):
        # Import here to make llama-cpp-python optional
        from llama_cpp import Llama

        self.registry = registry
        self.use_native_tools = use_native_tools
        self.temperature = temperature
        self.debug = debug
        self.minimal_contract = minimal_contract
        self._n_gpu_layers = n_gpu_layers  # Store for GPU detection

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

        # Build tools contract - minimal or full
        if minimal_contract:
            self.tools_contract = MINIMAL_CONTRACT_TEMPLATE.format(
                tool_list=registry.to_minimal_list()
            )
        else:
            self.tools_contract = TOOL_CONTRACT_TEMPLATE.format(
                registry_json=registry.to_registry_json()
            )

        if self.debug:
            print(f"\033[90m[DEBUG] Tools contract: {len(self.tools_contract)} chars\033[0m", file=sys.stderr)

        self.history: list[dict[str, Any]] = []
        self.history_trimmed: bool = False  # Flag to signal when history was trimmed
        self.partition = context_partition or ContextPartition()

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
        2. Partitioned context (summaries and/or transcript excerpts)
        3. Conversation history
        4. Any pending messages (current turn)

        Args:
            pending: Optional list of messages for the current turn
                    (e.g., user input and tool results not yet in history).

        Returns:
            Complete list of messages to send to the LLM.
        """
        messages = [self._build_system_message()]

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

        Args:
            name: Tool name or alias to execute.
            args: Dictionary of arguments to pass to the tool.

        Returns:
            Dictionary with name, arguments, and output.
            Errors are returned with an "error" key in output.
        """
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

    def chat_stream(self, user_input: str, max_tool_rounds: int = 3):
        """Stream chat response, yielding only the final response.

        Tool calls and tool results are added to history but NOT yielded.
        Only the final plain text response is yielded.

        Args:
            user_input: The user's message.
            max_tool_rounds: Maximum number of LLM generation rounds.

        Yields:
            str: The final response content (yielded as a single string).
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
            # Calculate message sizes for debug
            total_chars = sum(len(m.get("content", "")) for m in messages)
            self._debug_print(f"  build_messages: {(t_build - t_start)*1000:.2f}ms ({len(messages)} msgs, {total_chars} chars)")

            self.llm.reset()
            t_reset = time.perf_counter()
            self._debug_print(f"  llm.reset: {(t_reset - t_build)*1000:.2f}ms")

            # Use streaming to collect response
            if self.use_native_tools:
                stream = self.llm.create_chat_completion(
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    stream=True,
                )
            else:
                stream = self.llm.create_chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                )

            # Collect all tokens first (no streaming during collection)
            content = ""
            tool_calls_data = []
            t_stream_start = time.perf_counter()

            for chunk in stream:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                # Check for native tool calls
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

            t_stream_end = time.perf_counter()
            self._debug_print(f"  llm.generate: {(t_stream_end - t_stream_start)*1000:.2f}ms ({len(content)} chars)")

            # Check if it's a native tool call
            if tool_calls_data:
                self._debug_print(f"  detected: native tool call")
                # Native tool calls - add to history, execute, continue (no yield)
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

            # Try to parse as JSON contract format
            t_parse = time.perf_counter()
            parsed = self._parse_tool_call(content)
            self._debug_print(f"  parse_content_json: {(time.perf_counter() - t_parse)*1000:.2f}ms")

            if parsed and parsed["type"] == "tool_call":
                self._debug_print(f"  detected: contract tool call ({parsed.get('name')})")
                # Tool call in JSON format - add to history, execute, continue (no yield)
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
                    "role": "user",
                    "content": f"[Tool Result]: {output}",
                })
                self._debug_print(f"  format_result: {(time.perf_counter() - t_format)*1000:.2f}ms")
                continue

            else:
                self._debug_print(f"  detected: final response")
                # Final plain text response - add to history AND yield
                pending.append({"role": "assistant", "content": content})
                self.history.extend(pending)
                self._debug_print(f"  total round: {(time.perf_counter() - t_start)*1000:.2f}ms")
                yield content
                return

        # Max rounds reached
        self.history.extend(pending)
        yield "[Max tool rounds reached]"

    def chat(self, user_input: str, max_tool_rounds: int = 3) -> str:
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

        Returns:
            The final response string to show the user.

        Note:
            - Automatically resets model state before each generation (required
              for hybrid/Mamba models)
            - All messages (including tool results) are committed to history
            - History is trimmed if it exceeds context limit
        """
        # Trim history if needed (preserves system + pending space)
        self._trim_history()

        user_msg = {"role": "user", "content": user_input}
        pending: list[dict[str, Any]] = [user_msg]

        # Track tool calls to detect duplicates (prevent infinite loops)
        called_tools: set[str] = set()

        for round_num in range(max_tool_rounds):
            messages = self._build_messages(pending)

            # Reset model state (required for Mamba/SSM models like granite-hybrid)
            self.llm.reset()

            # Call LLM with or without native tools
            if self.use_native_tools:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
            else:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                )

            assistant_msg = response["choices"][0]["message"]
            content = assistant_msg.get("content", "")

            print(f"--- LLM Response (Round {round_num + 1}) ---")
            # Check for native tool calls first (if enabled)
            if self.use_native_tools and assistant_msg.get("tool_calls"):
                print("Detected native tool calls.")
                pending.append(assistant_msg)
                has_new_calls = False
                for tool_call in assistant_msg["tool_calls"]:
                    name = tool_call["function"]["name"]
                    args_json = tool_call["function"]["arguments"]
                    args = json.loads(args_json) if isinstance(args_json, str) else args_json

                    # Check for duplicate tool call (prevent infinite loops)
                    call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                    if call_key in called_tools:
                        # Duplicate call - return last result as final response
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
                    # All calls were duplicates - force exit
                    self.history.extend(pending)
                    return "[Tool already executed - please provide a response]"
                continue

            # Try to parse as our JSON contract format (fallback)
            parsed = self._parse_tool_call(content)

            if parsed and parsed["type"] == "tool_call":
                print("Detected tool call in JSON format.")
                # Tool call - execute and continue
                pending.append(assistant_msg)
                name = parsed.get("name", "")
                args = parsed.get("arguments", {})

                # Check for duplicate tool call (prevent infinite loops)
                call_key = f"{name}:{json.dumps(args, sort_keys=True)}"
                if call_key in called_tools:
                    # Duplicate call - force a final response with last result
                    self.history.extend(pending)
                    return "[Tool already executed - please provide a response]"
                called_tools.add(call_key)

                result = self._execute_tool_by_name(name, args)
                output = self._result2llm(result["output"])

                # Send tool result
                pending.append({
                    "role": "user",
                    "content": f"[Tool Result]: {output}",
                })
                continue

            else:
                print("No tool call detected - treating as final response.")
                # Plain text response (no JSON)
                pending.append(assistant_msg)
                self.history.extend(pending)
                return content

        # Max rounds - still commit what we have
        self.history.extend(pending)
        return "[Max tool rounds reached]"

    def clear(self):
        """Clear conversation history.

        Resets the conversation to a clean slate. System message
        (with tool contract) is automatically rebuilt on next chat() call.
        """
        self.history = []

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

        Returns:
            Dict with warmup info: {"tokens": N, "time_s": X}
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
        )
        elapsed = time.perf_counter() - t_start

        # Extract token count
        usage = response.get("usage", {})
        total_tokens = usage.get("total_tokens", 0) or usage.get("prompt_tokens", 0)

        # DO NOT add to history - this is warmup only
        return {
            "tokens": total_tokens,
            "time_s": elapsed,
        }
