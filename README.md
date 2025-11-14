# PDF Bind

This is a simple script to combine multiple PDFs into a single one and to create outline with each file being a chapter. 
It uses excellent [pikepdf](https://github.com/pikepdf/pikepdf) library to handle the PDFs.

## Installation
TODO: publish to pypi

This project uses [uv](https://github.com/astral-sh/uv) and [hatchling](https://github.com/pypa/hatch) for dependency management and packaging:


If needed, check the latest release number at https://github.com/SashaOv/pdfbind/releases/latest and replace `0.2.1` in the command below.

```bash
uv pip install https://github.com/SashaOv/pdfbind/releases/download/0.2.1/pdfbind-py3-none-any.whl
```

Of course, you can also install it using standard pip:

```bash
pip install https://github.com/SashaOv/pdfbind/releases/download/0.2.1/pdfbind-py3-none-any.whl
```

## Building


```bash
# Install development dependencies
uv pip install -e .
```

```bash
# Build the package
python -m build
```

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
pdfbind favorites.yaml
```

This command will create `favorites.pdf` in the same directory as the layout file.

## License

Licensed under the Apache License, Version 2.0 (the "License").

Copyright [2024] Sasha Ovsankin
