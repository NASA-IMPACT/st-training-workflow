# st-training-workflow

Fine-tunining workflows for Setence Transformer

# Setup

Use `uv`, `venv` and `pre-commit` hooks


## Set venv

```bash
uv init
uv venv --python 3.12
```

## Setup pre-commit hoooks

```bash
uv pip install pre-commit

pre-commit install
```

To install any additional dependencies use `uv add <package_name>`.
