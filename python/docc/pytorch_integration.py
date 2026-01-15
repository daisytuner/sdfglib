"""
Minimal Torch-MLIR Integration for Inference

This module provides lightweight integration for importing PyTorch models
into DOCC SDFGs via Torch-MLIR for inference only.

Design Philosophy:
- Minimal wrapper code for maintainability
- Focus on inference-only (no training wrappers)
- Let PyTorch handle training; DOCC handles optimized inference
- Map MLIR operations to SDFG library nodes
"""

from typing import Optional, Tuple


def import_pytorch_model(
    model, example_inputs: Tuple, export_path: Optional[str] = None
):
    """
    Import a PyTorch model for optimized inference with DOCC.

    This function traces a PyTorch model and converts it to an optimized SDFG
    for inference via Torch-MLIR.

    Args:
        model: A PyTorch nn.Module instance (in eval mode)
        example_inputs: Example inputs for tracing (required for shape inference)
        export_path: Optional path to save the SDFG

    Returns:
        Callable SDFG that can be used for inference

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> import docc
        >>>
        >>> class SimpleModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(10, 5)
        ...     def forward(self, x):
        ...         return self.linear(x)
        >>>
        >>> model = SimpleModel()
        >>> model.eval()
        >>> example_input = torch.randn(1, 10)
        >>> optimized_model = docc.import_pytorch_model(model, (example_input,))
    """
    try:
        from ._docc import mlir_import_pytorch

        return mlir_import_pytorch(model, example_inputs, export_path)
    except (ImportError, AttributeError):
        raise RuntimeError(
            "MLIR support not available. "
            "Please build with -DDOCC_BUILD_MLIR=ON to enable PyTorch model import."
        )


def import_torch_mlir(mlir_module_str: str, export_path: Optional[str] = None):
    """
    Import a Torch-MLIR module (as string) for optimized inference with DOCC.

    This provides a lower-level import path for users who already have MLIR IR.

    Args:
        mlir_module_str: MLIR module as a string (textual IR)
        export_path: Optional path to save the SDFG

    Returns:
        Callable SDFG that can be used for inference

    Example:
        >>> import docc
        >>>
        >>> mlir_str = '''
        ... module {
        ...   func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
        ...     return %arg0 : tensor<2x3xf32>
        ...   }
        ... }
        ... '''
        >>> optimized_model = docc.import_torch_mlir(mlir_str)
    """
    try:
        from ._docc import mlir_import_module

        return mlir_import_module(mlir_module_str, export_path)
    except (ImportError, AttributeError):
        raise RuntimeError(
            "MLIR support not available. "
            "Please build with -DDOCC_BUILD_MLIR=ON to enable Torch-MLIR import."
        )
