# PDF Bind

This is a simple script to combine multiple PDFs into a single one and to create outline with each file being a chapter. 
It uses excellent [pikepdf](https://github.com/pikepdf/pikepdf) library to handle the PDFs.

## Installation
TODO: publish to pypi

This project uses [uv](https://github.com/astral-sh/uv) and [hatchling](https://github.com/pypa/hatch) for dependency management and packaging:

### System Dependencies

For OCR functionality (indexing scanned PDFs), you'll need to install system dependencies:

**macOS:**
```bash
brew install poppler tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install poppler-utils tesseract-ocr
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install poppler-utils tesseract
```

**Windows:**
- Download Poppler from: https://github.com/oschwartz10612/poppler-windows/releases
- Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Add both to your system PATH

### Handwritten / Real Book charts

Some scanned Real Book volumes use a hand-lettered font that Tesseract struggles to read.  
`pdfbind` now ships with automatic handwriting fallbacks:

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) is tried first and handles most Real Book glyphs. The first run downloads ~300 MB of models/wheels.
- If PaddleOCR cannot read a page, we fall back to EasyOCR (~80 MB download) with aggressive spell-repair tuned for jazz titles.

Expect the first indexing run on a new machine to be slower while these models download, but subsequent runs are cached locally.

If needed, check the latest release number at https://github.com/SashaOv/pdfbind/releases/latest and replace `0.1.1` in the command below.

```bash
uv pip install https://github.com/SashaOv/pdfbind/releases/download/0.1.1/pdfbind-py3-none-any.whl
```

Of course, you can also install it using standard pip:

```bash
pip install https://github.com/SashaOv/pdfbind/releases/download/0.1.1/pdfbind-py3-none-any.whl
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
