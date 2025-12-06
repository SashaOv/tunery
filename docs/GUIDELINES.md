# Repository Guidelines

## Project Structure & Module Organization
- We are using UV and pytest
- The code structure is: `tunery/` is the core source, `tests/` is where the test go

## Coding Style
- Avoid noise and incorrect behavior. ALWAYS check comments and documentation to be in sync with the code.
- DRY. Factor out common code patterns into functions. 
- Where possible, derive coding convensions from the code itself.
- AVOID hardcoded file paths.

## Python Style & Naming Conventions
- Target Python 3.12 features (e.g., `typing.Sequence`, union syntax); use 4-space indents and keep modules import-order clean (`stdlib`, `third-party`, `local`).
- Favor `Path` objects for filesystem work, snake_case for functions and variables, and CapitalizedClass for types.
- Favor using types from `typing` package.
- Favor using @dataclass or Pydantic models instead of dictionaries.
- Prefer canonic Python file structure: imports -> variables and classes -> functions.

## Testing Guidelines
- Write pytest functions named `test_<behavior>` inside files that mirror the module under test; parameterize when covering multiple layout shapes.
- Use tiny synthetic PDFs checked into `tests/fixtures/` or generate them on the fly to avoid large binaries; clean up temporary outputs in `tmp_path`.
- Aim for coverage around edge cases: partial page ranges (`page`/`length`), nested sections, and invalid YAML entries.

## Commit & Pull Request Guidelines
- Follow the existing imperative, present-tense commit style (`Change build system to UV`, `Fix build name`); keep subject lines under ~60 characters and explain motivation plus impact in the body when needed.
- For pull requests include: summary of changes, linked issues, results of test execution, and screenshots or sample YAML if UX changes affect the CLI output.
- Ensure CI (or local pytest) passes before requesting review, and mention any remaining TODOs or follow-ups explicitly.
