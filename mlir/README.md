# MLIR (PyTorch/torch-mlir)

The MLIR module contains Python bindings and a MLIR conversion for JIT compiling PyTorch models to SDFGs.

## Setup

For compatibility reasons with torch-mlir Python Version 3.11 is required. The following command initializes a virtual environment.
```bash
python3.11 -m venv mlir_venv
source mlir_venv/bin/activate
```

Install the package dependencies, as follows.
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
