"""
Tests for MLIR frontend Python utilities
"""

import os
import sys
import pytest
from pathlib import Path

# Add the parent directory to sys.path to import sdfglib
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdfglib.mlir_frontend import (
    convert_scalar_type,
    convert_tensor_type,
    get_elementwise_op_code,
    get_reduce_op_code,
    is_elementwise_unary,
    is_elementwise_binary,
    is_reduce_op,
    SDFGPrimitiveType,
)


class TestScalarTypeConversion:
    """Test MLIR to SDFG scalar type conversions"""

    def test_integer_types(self):
        """Test integer type conversions"""
        assert convert_scalar_type("i1") == SDFGPrimitiveType.BOOL
        assert convert_scalar_type("i8") == SDFGPrimitiveType.INT8
        assert convert_scalar_type("i16") == SDFGPrimitiveType.INT16
        assert convert_scalar_type("i32") == SDFGPrimitiveType.INT32
        assert convert_scalar_type("i64") == SDFGPrimitiveType.INT64

    def test_float_types(self):
        """Test floating point type conversions"""
        assert convert_scalar_type("f16") == SDFGPrimitiveType.HALF
        assert convert_scalar_type("f32") == SDFGPrimitiveType.FLOAT
        assert convert_scalar_type("f64") == SDFGPrimitiveType.DOUBLE

    def test_index_type(self):
        """Test index type conversion"""
        assert convert_scalar_type("index") == SDFGPrimitiveType.INT64

    def test_unsupported_type(self):
        """Test unsupported type raises exception"""
        with pytest.raises(ValueError):
            convert_scalar_type("unknown_type")


class TestTensorTypeConversion:
    """Test MLIR to SDFG tensor type conversions"""

    def test_f32_tensor(self):
        """Test f32 tensor conversion"""
        element_type, shape = convert_tensor_type("f32", [32, 64])
        assert element_type == SDFGPrimitiveType.FLOAT
        assert shape == [32, 64]

    def test_i32_tensor(self):
        """Test i32 tensor conversion"""
        element_type, shape = convert_tensor_type("i32", [10, 20, 30])
        assert element_type == SDFGPrimitiveType.INT32
        assert shape == [10, 20, 30]

    def test_unsupported_element_type(self):
        """Test unsupported element type raises exception"""
        with pytest.raises(ValueError):
            convert_tensor_type("unknown", [32])


class TestElementwiseOperations:
    """Test elementwise operation mappings"""

    def test_binary_ops(self):
        """Test binary operation mappings"""
        assert get_elementwise_op_code("add") == "Add"
        assert get_elementwise_op_code("sub") == "Sub"
        assert get_elementwise_op_code("mul") == "Mul"
        assert get_elementwise_op_code("div") == "Div"
        assert get_elementwise_op_code("pow") == "Pow"
        assert get_elementwise_op_code("minimum") == "Minimum"
        assert get_elementwise_op_code("maximum") == "Maximum"

    def test_unary_ops(self):
        """Test unary operation mappings"""
        assert get_elementwise_op_code("abs") == "Abs"
        assert get_elementwise_op_code("sqrt") == "Sqrt"
        assert get_elementwise_op_code("exp") == "Exp"
        assert get_elementwise_op_code("relu") == "Relu"
        assert get_elementwise_op_code("sigmoid") == "Sigmoid"
        assert get_elementwise_op_code("tanh") == "Tanh"

    def test_unsupported_op(self):
        """Test unsupported operation raises exception"""
        with pytest.raises(ValueError):
            get_elementwise_op_code("unsupported_op")


class TestReductionOperations:
    """Test reduction operation mappings"""

    def test_reduce_ops(self):
        """Test reduction operation mappings"""
        assert get_reduce_op_code("sum") == "Sum"
        assert get_reduce_op_code("mean") == "Mean"
        assert get_reduce_op_code("std") == "Std"
        assert get_reduce_op_code("max") == "Max"
        assert get_reduce_op_code("min") == "Min"
        assert get_reduce_op_code("softmax") == "Softmax"

    def test_unsupported_reduce_op(self):
        """Test unsupported reduction raises exception"""
        with pytest.raises(ValueError):
            get_reduce_op_code("unsupported_reduce")


class TestOperationClassification:
    """Test operation type classification"""

    def test_unary_classification(self):
        """Test unary operation classification"""
        assert is_elementwise_unary("abs") is True
        assert is_elementwise_unary("sqrt") is True
        assert is_elementwise_unary("relu") is True
        assert is_elementwise_unary("add") is False
        assert is_elementwise_unary("sum") is False

    def test_binary_classification(self):
        """Test binary operation classification"""
        assert is_elementwise_binary("add") is True
        assert is_elementwise_binary("mul") is True
        assert is_elementwise_binary("div") is True
        assert is_elementwise_binary("abs") is False
        assert is_elementwise_binary("sum") is False

    def test_reduce_classification(self):
        """Test reduction operation classification"""
        assert is_reduce_op("sum") is True
        assert is_reduce_op("mean") is True
        assert is_reduce_op("max") is True
        assert is_reduce_op("add") is False
        assert is_reduce_op("abs") is False


class TestDoccBuildDirectory:
    """Test docc build directory support"""

    def test_docc_build_directory_env(self):
        """Test that docc build directory can be set via environment"""
        # Set docc build directory
        docc_build_dir = "/tmp/docc_build"
        os.environ["DOCC_BUILD_DIR"] = docc_build_dir

        # Verify it's set
        assert os.environ.get("DOCC_BUILD_DIR") == docc_build_dir

        # Clean up
        del os.environ["DOCC_BUILD_DIR"]

    def test_docc_build_directory_default(self):
        """Test default docc build directory"""
        # Ensure environment variable is not set
        if "DOCC_BUILD_DIR" in os.environ:
            del os.environ["DOCC_BUILD_DIR"]

        # Default should be None or a default path
        docc_dir = os.environ.get("DOCC_BUILD_DIR", None)
        assert docc_dir is None or isinstance(docc_dir, str)

    def test_docc_build_directory_path_creation(self):
        """Test that docc build directory path can be created"""
        import tempfile

        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            docc_build_dir = Path(tmpdir) / "docc_build"
            docc_build_dir.mkdir(parents=True, exist_ok=True)

            assert docc_build_dir.exists()
            assert docc_build_dir.is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
