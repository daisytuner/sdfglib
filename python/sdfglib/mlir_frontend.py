"""
MLIR Frontend utilities for Python

This module provides Python wrappers for MLIR-style type conversions
and operation mappings to SDFG library nodes.
"""

from typing import List, Dict, Tuple
from enum import Enum


class MLIRScalarType(Enum):
    """MLIR scalar types that can be converted to SDFG types"""

    I1 = "i1"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    F16 = "f16"
    F32 = "f32"
    F64 = "f64"
    INDEX = "index"


class SDFGPrimitiveType(Enum):
    """SDFG primitive types"""

    BOOL = "bool"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    HALF = "half"
    FLOAT = "float"
    DOUBLE = "double"


# Mapping from MLIR types to SDFG types
MLIR_TO_SDFG_TYPE_MAP: Dict[str, SDFGPrimitiveType] = {
    "i1": SDFGPrimitiveType.BOOL,
    "i8": SDFGPrimitiveType.INT8,
    "i16": SDFGPrimitiveType.INT16,
    "i32": SDFGPrimitiveType.INT32,
    "i64": SDFGPrimitiveType.INT64,
    "f16": SDFGPrimitiveType.HALF,
    "f32": SDFGPrimitiveType.FLOAT,
    "f64": SDFGPrimitiveType.DOUBLE,
    "index": SDFGPrimitiveType.INT64,
}

# Elementwise operation mappings
ELEMENTWISE_OPS: Dict[str, str] = {
    # Binary operations
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "div": "Div",
    "pow": "Pow",
    "minimum": "Minimum",
    "maximum": "Maximum",
    # Unary operations
    "abs": "Abs",
    "sqrt": "Sqrt",
    "exp": "Exp",
    "erf": "Erf",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
    "relu": "Relu",
    "leaky_relu": "LeakyRelu",
    "elu": "Elu",
    "hard_sigmoid": "HardSigmoid",
    "cast": "Cast",
}

# Reduction operation mappings
REDUCE_OPS: Dict[str, str] = {
    "sum": "Sum",
    "mean": "Mean",
    "std": "Std",
    "max": "Max",
    "min": "Min",
    "softmax": "Softmax",
}


def convert_scalar_type(mlir_type: str) -> SDFGPrimitiveType:
    """
    Convert MLIR scalar type string to SDFG primitive type

    Args:
        mlir_type: MLIR type string (e.g., "f32", "i64")

    Returns:
        SDFG primitive type

    Raises:
        ValueError: If the MLIR type is not supported
    """
    if mlir_type not in MLIR_TO_SDFG_TYPE_MAP:
        raise ValueError(f"Unsupported MLIR scalar type: {mlir_type}")
    return MLIR_TO_SDFG_TYPE_MAP[mlir_type]


def convert_tensor_type(element_type: str, shape: List[int]) -> Tuple[SDFGPrimitiveType, List[int]]:
    """
    Convert MLIR tensor type to SDFG flat pointer type

    In SDFG, tensors are represented as flat pointers to scalars.
    The shape information is returned separately.

    Args:
        element_type: Element type string (e.g., "f32")
        shape: Tensor shape dimensions

    Returns:
        Tuple of (element primitive type, shape)

    Raises:
        ValueError: If the element type is not supported
    """
    sdfg_type = convert_scalar_type(element_type)
    return (sdfg_type, shape)


def get_elementwise_op_code(op_name: str) -> str:
    """
    Get SDFG library node code for elementwise operation

    Args:
        op_name: MLIR operation name

    Returns:
        SDFG library node code string

    Raises:
        ValueError: If the operation is not supported
    """
    if op_name not in ELEMENTWISE_OPS:
        raise ValueError(f"Unsupported elementwise operation: {op_name}")
    return ELEMENTWISE_OPS[op_name]


def get_reduce_op_code(op_name: str) -> str:
    """
    Get SDFG library node code for reduction operation

    Args:
        op_name: MLIR operation name

    Returns:
        SDFG library node code string

    Raises:
        ValueError: If the operation is not supported
    """
    if op_name not in REDUCE_OPS:
        raise ValueError(f"Unsupported reduce operation: {op_name}")
    return REDUCE_OPS[op_name]


def is_elementwise_unary(op_name: str) -> bool:
    """Check if operation is elementwise unary"""
    unary_ops = {
        "abs",
        "sqrt",
        "exp",
        "erf",
        "sigmoid",
        "tanh",
        "relu",
        "leaky_relu",
        "elu",
        "hard_sigmoid",
        "cast",
    }
    return op_name in unary_ops


def is_elementwise_binary(op_name: str) -> bool:
    """Check if operation is elementwise binary"""
    binary_ops = {"add", "sub", "mul", "div", "pow", "minimum", "maximum"}
    return op_name in binary_ops


def is_reduce_op(op_name: str) -> bool:
    """Check if operation is a reduction"""
    return op_name in REDUCE_OPS
