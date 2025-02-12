# PDF Bind

This is a simple script to combine multiple PDFs into a single one and to create outline with each file being a chapter. 
It uses excellent [pikepdf](https://github.com/pikepdf/pikepdf) library to handle the PDFs.

## Installation
TODO: publish to pypi

You can install a release using [pipx](https://github.com/pypa/pipx):

If needed, check the latest release number at https://github.com/SashaOv/pdfbind/releases/latest and replace `0.0.1` in the command below.

```bash
pipx install https://github.com/SashaOv/pdfbind/releases/download/0.0.1/pdfbind-py3-none-any.whl


Of course, you can also install it using standard pip:

```bash
pip install https://github.com/SashaOv/pdfbind/releases/download/0.0.1/pdfbind-py3-none-any.whl
```




## Buildling

This is a standard Poetry project, so all the usual commands apply:

```bash
poetry install
```

```bash
poetry build
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

Licensed under the Apache License, Version 2.0 (the "License"). See the [LICENSE](LICENSE) file for more details.

Copyright [2024] Sasha Ovsankin
