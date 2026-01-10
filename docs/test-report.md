# Test Report

**Date:** 2026-01-06
**Updated:** 2026-01-07
**Project:** LLM Tool-Calling System (l3m-backend)
**Test Framework:** pytest 9.0.2
**Python Version:** 3.12.11

> **Note:** This report documents the initial test suite. The project has since been reorganized
> into a pip-installable package structure under `src/l3m_backend/`. Current test count: 400 tests.

---

## Summary

| Metric      | Value |
|-------------|-------|
| Total Tests | 400   |
| Passed      | 400   |
| Failed      | 0     |
| Skipped     | 0     |
| Duration    | 1.84s |
| Coverage    | 100%  |

**Status: ALL TESTS PASSING**

---

## Coverage Report

| File     | Statements | Missed | Coverage |
|----------|------------|--------|----------|
| tools.py | 48         | 0      | 100%     |

---

## Test Results by Category

### 1. Weather Tool Tests (6 tests)

| Test                                       | Status | Description                            |
|--------------------------------------------|--------|----------------------------------------|
| `test_get_weather_celsius_success`         | PASSED | Valid city with celsius (default unit) |
| `test_get_weather_fahrenheit_success`      | PASSED | Valid city with fahrenheit unit        |
| `test_get_weather_city_not_found`          | PASSED | Non-existent city handling             |
| `test_get_weather_network_error`           | PASSED | Network error handling                 |
| `test_get_weather_timeout`                 | PASSED | Timeout error handling                 |
| `test_get_weather_registered_with_aliases` | PASSED | Verifies aliases (weather, w)          |

### 2. Weather Helper Functions (3 tests)

| Test                                | Status | Description                               |
|-------------------------------------|--------|-------------------------------------------|
| `test_geocode_success`              | PASSED | Geocoding API returns correct coordinates |
| `test_geocode_city_not_found`       | PASSED | ValueError raised for invalid city        |
| `test_get_weather_code_description` | PASSED | WMO weather code mapping                  |

### 3. Calculator Tool Tests (13 tests)

| Test                                     | Status | Description                         |
|------------------------------------------|--------|-------------------------------------|
| `test_calculate_addition`                | PASSED | Basic addition                      |
| `test_calculate_subtraction`             | PASSED | Basic subtraction                   |
| `test_calculate_multiplication`          | PASSED | Basic multiplication                |
| `test_calculate_division`                | PASSED | Basic division                      |
| `test_calculate_complex_expression`      | PASSED | Complex nested expressions          |
| `test_calculate_float_operations`        | PASSED | Floating point operations           |
| `test_calculate_division_by_zero`        | PASSED | Error handling for division by zero |
| `test_calculate_invalid_characters`      | PASSED | Security: blocks code injection     |
| `test_calculate_letters_rejected`        | PASSED | Rejects non-numeric characters      |
| `test_calculate_empty_expression`        | PASSED | Edge case: empty string             |
| `test_calculate_only_whitespace`         | PASSED | Edge case: whitespace only          |
| `test_calculate_parentheses_only`        | PASSED | Edge case: empty parentheses        |
| `test_calculate_registered_with_aliases` | PASSED | Verifies aliases (calc, c)          |

### 4. Time Tool Tests (4 tests)

| Test                                    | Status | Description                                   |
|-----------------------------------------|--------|-----------------------------------------------|
| `test_get_time_format`                  | PASSED | Correct datetime format (YYYY-MM-DD HH:MM:SS) |
| `test_get_time_reasonable_value`        | PASSED | Returns current time (within 2 seconds)       |
| `test_get_time_llm_format`              | PASSED | Correct LLM format output                     |
| `test_get_time_registered_with_aliases` | PASSED | Verifies aliases (time, t)                    |

### 5. ToolOutput Wrapper Tests (6 tests)

| Test                                      | Status | Description                 |
|-------------------------------------------|--------|-----------------------------|
| `test_tool_output_structure`              | PASSED | Has all required fields     |
| `test_tool_output_type_name`              | PASSED | Correct type_name attribute |
| `test_tool_output_id_unique`              | PASSED | Generates unique IDs        |
| `test_tool_output_string_format_template` | PASSED | String template formatting  |
| `test_tool_output_lambda_format`          | PASSED | Lambda/callable formatting  |
| `test_tool_output_gui_format_none`        | PASSED | GUI format can be None      |

### 6. Registry Integration Tests (5 tests)

| Test                        | Status | Description                 |
|-----------------------------|--------|-----------------------------|
| `test_all_tools_registered` | PASSED | All 3 tools registered      |
| `test_execute_via_registry` | PASSED | Execute tool via registry   |
| `test_execute_via_alias`    | PASSED | Execute tool via alias      |
| `test_execute_call_dict`    | PASSED | Execute from tool_call dict |
| `test_tool_to_openai_spec`  | PASSED | Generate valid OpenAI specs |

### 7. Edge Cases and Error Handling (6 tests)

| Test                                           | Status | Description                          |
|------------------------------------------------|--------|--------------------------------------|
| `test_weather_with_special_characters_in_city` | PASSED | Unicode characters (e.g., Sao Paulo) |
| `test_calculate_with_nested_parentheses`       | PASSED | Deeply nested expressions            |
| `test_calculate_negative_numbers`              | PASSED | Negative number handling             |
| `test_calculate_very_long_expression`          | PASSED | Very long expressions (100 terms)    |
| `test_weather_http_error`                      | PASSED | HTTP error handling                  |
| `test_time_multiple_calls_different`           | PASSED | Time progresses between calls        |

---

## Test Commands

```bash
# Run all tests
~/.venvs/l3m/bin/python -m pytest tests/ -v

# Run tool tests
~/.venvs/l3m/bin/python -m pytest tests/test_tools.py -v

# Run with coverage
~/.venvs/l3m/bin/python -m pytest tests/test_tools.py --cov=l3m_backend.tools --cov-report=term-missing

# Run specific test class
~/.venvs/l3m/bin/python -m pytest tests/test_tools.py::TestCalculatorTool -v

# Run specific test
~/.venvs/l3m/bin/python -m pytest tests/test_tools.py::TestCalculatorTool::test_calculate_invalid_characters -v
```

---

## Notes

- All API calls to Open-Meteo are mocked to ensure tests run offline and consistently
- Security tests verify that code injection via `eval()` is blocked by the character allowlist
- ToolOutput wrapper tests verify integration with the backend registry system
