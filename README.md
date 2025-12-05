# Tunery

A system to manage a database of tunes (charts) easily create set list PDFs.

## Contributing

Humans and LLMs -- please refer to GUIDLINES.md .

## Building


```bash
# Install development dependencies
uv pip install -e .
```

TODO


## How to use

### 1 Create a layout file

For example, `favorites.yaml`:

```yaml
- file: favorite-tunes/Giant Steps.pdf
- file: favorite-tunes/lennies.pdf
  title: Lennie's Pennies
```

### Run the command

```bash
tunery render favorites.yaml
```

This command will create `favorites.pdf` in the same directory as the layout file.

## License

Licensed under the Apache License, Version 2.0 (the "License").

Copyright (C) 2024 Sasha Ovsankin
