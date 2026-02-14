import inspect
import numpy as np

from typing import get_origin, get_args
from docc.sdfg import (
    PrimitiveType,
    Scalar,
    Pointer,
    Array,
    Structure,
    Tensor,
    Type,
)


def _scalar_from_numpy_dtype(dtype):
    # Handle np.dtype[ScalarType] generic annotation
    if get_origin(dtype) is np.dtype:
        args = get_args(dtype)
        if args:
            dtype = args[0]

    # Handle np.dtype instances
    if isinstance(dtype, np.dtype):
        dtype = dtype.type

    # Map to primitive type
    if dtype is float or dtype is np.float64:
        return Scalar(PrimitiveType.Double)
    elif dtype is np.float32:
        return Scalar(PrimitiveType.Float)
    elif dtype is bool or dtype is np.bool_:
        return Scalar(PrimitiveType.Bool)
    elif dtype is int or dtype is np.int64:
        return Scalar(PrimitiveType.Int64)
    elif dtype is np.int32:
        return Scalar(PrimitiveType.Int32)
    elif dtype is np.int16:
        return Scalar(PrimitiveType.Int16)
    elif dtype is np.int8:
        return Scalar(PrimitiveType.Int8)
    elif dtype is np.uint64:
        return Scalar(PrimitiveType.UInt64)
    elif dtype is np.uint32:
        return Scalar(PrimitiveType.UInt32)
    elif dtype is np.uint16:
        return Scalar(PrimitiveType.UInt16)
    elif dtype is np.uint8:
        return Scalar(PrimitiveType.UInt8)

    raise ValueError(f"Unsupported numpy dtype: {dtype}")


def sdfg_type_from_dtype(dtype):
    if isinstance(dtype, Type):
        return dtype

    # Handle numpy.ndarray[Shape, DType] type annotations
    if get_origin(dtype) is np.ndarray:
        args = get_args(dtype)
        if len(args) >= 2:
            elem_type = _scalar_from_numpy_dtype(args[1])
            return Pointer(elem_type)
        # Unparameterized ndarray defaults to void pointer
        return Pointer(Scalar(PrimitiveType.Void))

    # Handle np.dtype[ScalarType] annotations
    if get_origin(dtype) is np.dtype:
        return _scalar_from_numpy_dtype(dtype)

    if dtype is float or dtype is np.float64:
        return Scalar(PrimitiveType.Double)
    elif dtype is np.float32:
        return Scalar(PrimitiveType.Float)
    elif dtype is bool or dtype is np.bool_:
        return Scalar(PrimitiveType.Bool)
    elif dtype is int or dtype is np.int64:
        return Scalar(PrimitiveType.Int64)
    elif dtype is np.int32:
        return Scalar(PrimitiveType.Int32)
    elif dtype is np.int16:
        return Scalar(PrimitiveType.Int16)
    elif dtype is np.int8:
        return Scalar(PrimitiveType.Int8)
    elif dtype is np.uint64:
        return Scalar(PrimitiveType.UInt64)
    elif dtype is np.uint32:
        return Scalar(PrimitiveType.UInt32)
    elif dtype is np.uint16:
        return Scalar(PrimitiveType.UInt16)
    elif dtype is np.uint8:
        return Scalar(PrimitiveType.UInt8)

    # Handle Python classes - map to Structure type
    if inspect.isclass(dtype):
        return Pointer(Structure(dtype.__name__))

    raise ValueError(f"Cannot map type to SDFG type: {dtype}")
