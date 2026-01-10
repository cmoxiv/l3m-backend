"""
Data models for the tool registry.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from l3m_backend.core.helpers import _create_params_model


class ToolOutput(BaseModel):
    """Structured tool output."""
    type_name: str
    id: str
    data: Any
    llm_format: str | None = None
    gui_format: str | None = None

    @classmethod
    def create(
        cls,
        output: Any,
        llm_format: str | None = None,
        gui_format: str | None = None,
    ) -> ToolOutput:
        return cls(
            type_name=type(output).__name__,
            id=f"{type(output).__name__}_{uuid4().hex}",
            data=output,
            llm_format=llm_format,
            gui_format=gui_format,
        )


class ToolResult(BaseModel):
    """Result of a tool execution."""
    name: str
    arguments: dict[str, Any]
    output: Any


class ToolEntry(BaseModel):
    """Registry entry for a single tool."""
    name: str
    callable_fn: Callable = Field(exclude=True)
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    params_model: type[BaseModel] | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    def get_params_model(self) -> type[BaseModel]:
        """Get or create the Pydantic model for this tool's parameters."""
        if self.params_model is not None:
            return self.params_model
        self.params_model = _create_params_model(self.callable_fn, self.name)
        return self.params_model

    def get_description(self) -> str:
        """Get description from override or docstring."""
        if self.description:
            return self.description
        doc = inspect.getdoc(self.callable_fn) or ""
        return doc.split("\n\n", 1)[0].strip()

    def to_openai_spec(self) -> dict[str, Any]:
        """Convert to OpenAI-style tool specification."""
        params_model = self.get_params_model()
        schema = params_model.model_json_schema()

        # Clean up Pydantic's schema output
        schema.pop("title", None)
        schema.pop("$defs", None)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        schema.setdefault("required", [])
        schema["additionalProperties"] = False

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.get_description(),
                "parameters": schema,
            },
        }
