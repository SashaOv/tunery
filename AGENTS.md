# Tunery

## Gate

Run before every commit; both must pass:

```sh
uv run ruff check tunery tests
uv run pytest -q
```

Ruff enforces imports at the top of the module (`PLC0415`); move an import
inline only when logic requires it, with a comment saying why.

## Specs

- `docs/SPEC.md`: current behavior.
- `docs/todo/*.md`: change specs. `library-based.md` carries the LIB builds;
  each patch is one commit whose subject starts with its name
  (`LIB5.2: ...`).

## Tests

Fixtures build libraries as blank PDFs under pytest's `tmp_path`, named
after real library files. Tests never depend on absolute paths or the
author's drives.
