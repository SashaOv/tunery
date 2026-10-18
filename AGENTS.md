# Tunery

## Gate

Run `uv run build.py gate` before every commit; do not commit unless it
passes. It lints and runs the tests. 

## Builds

The behavior spec is `docs/SPEC.md`; change specs live in `docs/todo/`.
Each patch is one commit whose subject starts with its name (`LIB5.2: ...`).

## Tests

Fixtures build libraries as blank PDFs under pytest's `tmp_path`, named
after real library files (`tests/helpers.py`, `tests/data/`). Tests never
depend on absolute paths or the author's drives.

# Python Guidelines

Honor system-wide guidelines.

- Move an import inline only when logic requires it, with a comment saying why.