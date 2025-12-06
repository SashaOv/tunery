# Tunery

A system to manage a database of tunes (charts) easily create set list PDFs.

## Usage
Two main functions: 1) build index, 2) render the PDF file based on the layout YAML file.

For more information, do `tunery --help`

### Example layout file:

`favorites.yaml`:

```yaml
- file: favorite-tunes/Giant Steps.pdf
- file: favorite-tunes/lennies.pdf
  title: Lennie's Pennies
```

## Developing

**Guidelines**: Humans and LLMs -- please follow @docs/GUIDLINES.md . 

### Set up

- Install [UV](https://docs.astral.sh/uv/getting-started/installation/)
- Create virtual environment: `uv venv`
- Install dependencies (including development dependencies) `uv sync`
- (Optional) Activate virtual environment: `source .venv/bin/activate`


## License

Licensed under the Apache License, Version 2.0 (the "License").

Copyright (C) 2024-2025 Sasha Ovsankin
