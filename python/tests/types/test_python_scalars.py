import pytest

from docc.sdfg import (
    PrimitiveType,
    Scalar,
    Pointer,
    Structure,
    Tensor,
)
from docc.python.types import sdfg_type_from_dtype


def test_python_float():
    result = sdfg_type_from_dtype(float)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.Double


def test_python_int():
    result = sdfg_type_from_dtype(int)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.Int64


def test_python_bool():
    result = sdfg_type_from_dtype(bool)
    assert isinstance(result, Scalar)
    assert result.primitive_type == PrimitiveType.Bool
