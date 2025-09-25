## Purpose

These instructions translate the project's Pull Request checklist into explicit guidance for GitHub Copilot / code generation helpers. Use them when proposing code, docstrings, tests, or PR descriptions.

## Checklist (requirements to satisfy)
- Documentation: All new or substantially edited functions must have documentation in the project's existing style (see examples below).
- Comments: Add explanatory code comments where necessary; focus on *why* something is done rather than *what*.
- Type annotations: All added or substantially edited functions must include argument and return type annotations.
- Type annotations II: Use type annotations for variables within functions.
- Type annotations III: Use `|` for union types (e.g., `Optional[float]` becomes `float | None`).
- Variable naming: Use clear, descriptive names understandable by a non-domain expert.
- Units: Include units for parameters and return values unless they are standard SI defaults (e.g., hydrology in meters).
- Monetary units: All monetary amounts are nominal USD for the stated year; indicate the year when appropriate.

## How Copilot should generate or modify code

- Docstrings: Generate Google-style docstrings for every new or substantially edited function. Include `Args`, `Returns`, and `Raises` sections. If units apply, put them in the parameter or return description.
- Types: Add Python type hints to function signatures (PEP 484). Use `|` where appropriate, and prefer concrete types (e.g., `float`, `int`, `str`, `List[float]`, `Dict[str, Any]`).
- Comments: Insert short inline comments to explain *why* a non-obvious step is needed. Avoid comments that restate the code.
- Names: Prefer descriptive names (e.g., `drainage_rate_m_per_s` instead of `r`) and avoid domain jargon when a plain name will do. If domain terms are needed, add a brief docstring note explaining them.
- Units and money: Always document units in parameter/return docstrings, e.g., `(meters)`, `(USD, nominal)`. If a function converts units or uses a specific year's dollars, state that clearly.
- Errors: Add input validation and raise explicit exceptions for invalid inputs. Use `assert` only for internal invariants, not for validating user input.
- Tests: When generating code that changes behavior, produce or update unit tests (pytest) covering the happy path and at least one edge case (invalid input or boundary). Prefer to add tests under `tests/` following the project's existing test naming patterns.
- Do not add extensive "-------" or similar in comments.

## Docstring template (Google style)

Use this template for new functions and substantial edits (do not include types in the docstring, as they are already in the function signature):

"""
Short one-line description of the function.

Longer description if necessary.

Notes:
    Important notes about the function's behavior, assumptions, or edge cases.

Args:
	param_name: Description of param_name (meters). Explain units here.
	other_param: Description. State any valid ranges.

Returns:
	Description of the return value (m).

Raises:
	ValueError: If input is invalid.

"""

## PR / Commit message guidance

- In the PR description, list which checklist items you satisfied and reference updated test files.
- If behavior changed, include a short note on migration or user-facing impact (units, default values, monetary-year assumptions).

## Minimal examples Copilot should follow

- When renaming variables, ensure callsites are updated and tests still pass.

