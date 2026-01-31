from typing import Any
import torch.nn as nn
import torch

from torch_mlir import fx

from ._docc_ml import MLIRModule


def import_from_pytorch(
    model: nn.Module,
    example_input: Any,
) -> str:
    # Import a PyTorch model to MLIR using torch-mlir FX
    torch_mlir = fx.export_and_import(
        model, example_input, output_type="linalg_on_tensors"
    )
    torch_mlir = str(torch_mlir)

    # Convert to SDFG dialect
    mlir_module = MLIRModule(torch_mlir)
    mlir_module.convert_to_sdfg()

    return mlir_module.to_string()
