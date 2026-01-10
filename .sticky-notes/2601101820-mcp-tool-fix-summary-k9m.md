# MCP Tool Fix Summary

Created: 2026-01-10 18:20

---

## Problem

MCP tools were registered but returned empty results. The LLM would call the tool but get nothing back.

## Root Causes & Fixes

### 1. Params Model Issue

**Problem**: MCP tool callables use `**kwargs`, so `_create_params_model()` generated an empty schema. When `model_validate()` ran, all arguments were dropped.

**Fix**: Create a proper Pydantic model from the MCP tool's `inputSchema` (JSON Schema):

```python
def _create_params_model_from_schema(name: str, schema: dict) -> type[BaseModel]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    # ... build field_definitions from properties
    return create_model(model_name, **field_definitions)
```

### 2. Name Normalization Issue

**Problem**: Tools registered with dots (`mcp.test.score`) but `registry.get()` normalizes names (`mcp_test_score`), causing lookup failures.

**Fix**: Use `_normalize_name()` when registering MCP tools:

```python
normalized_name = _normalize_name(full_name)
self.registry._tools[normalized_name] = entry
```

## Files Modified

- `src/l3m_backend/mcp/client/registry_adapter.py`

## Verification

```python
result = registry.execute('mcp.test.score', {'n': 123.0, 'b': 5.0})
# Returns: 2.9899782515341085
```
