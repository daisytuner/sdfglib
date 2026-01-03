# MLIR Frontend Implementation Summary

## Overview

This implementation adds a minimal MLIR-style frontend to sdfglib for converting torch-mlir types to SDFG types and mapping typical ML operations to existing library nodes.

## What Was Implemented

### 1. C++ MLIR Frontend (`include/sdfg/frontend/mlir_frontend.h`, `src/frontend/mlir_frontend.cpp`)

#### Type Conversions
- **Scalar Types**: Converts MLIR scalar types to SDFG scalar types
  - Integer types: `i1`, `i8`, `i16`, `i32`, `i64`
  - Floating point types: `f16`, `f32`, `f64`
  - Index type: `index` (maps to `i64`)
  
- **Tensor Types**: Converts MLIR tensor types to SDFG flat pointers
  - `tensor<NxMx...xT>` → `Pointer(Scalar(T))`
  - Shape information is extracted separately for use with library nodes
  - Aligns with SDFG's design where tensor library nodes expect flat pointers with linearized indexing

#### Operation Mappings
- **Elementwise Operations**: Maps MLIR ops to existing elementwise library nodes
  - Binary: `add`, `sub`, `mul`, `div`, `pow`, `minimum`, `maximum`
  - Unary: `abs`, `sqrt`, `exp`, `erf`, `sigmoid`, `tanh`, `relu`, `leaky_relu`, `elu`, `hard_sigmoid`, `cast`
  
- **Reduction Operations**: Maps MLIR ops to existing reduce library nodes
  - `sum`, `mean`, `std`, `max`, `min`, `softmax`

### 2. Python Frontend (`python/sdfglib/mlir_frontend.py`)

A Python module providing the same functionality as the C++ frontend:
- Type conversion utilities (scalar and tensor)
- Operation mapping dictionaries
- Helper functions for operation classification

### 3. Tests

#### C++ Tests (`tests/frontend/mlir_frontend_test.cpp`)
- Scalar type conversion tests
- Tensor type conversion tests
- Shape conversion tests
- Operation mapping tests
- Operation classification tests
- **Status**: All 7 tests passing

#### Python Tests (`python/tests/test_mlir_frontend.py`)
- Scalar and tensor type conversion tests
- Operation mapping tests
- Docc build directory support tests
- **Status**: All 18 tests passing

### 4. Integration

- Updated `CMakeLists.txt` to include new frontend source file
- Updated `tests/CMakeLists.txt` to include frontend tests
- Updated `.gitignore` for Python cache files
- Formatted all code with `clang-format` and `black`

## How It Works

### Type Conversion Flow

1. **Scalar Types**: Direct mapping from MLIR type strings to SDFG `PrimitiveType`
   ```cpp
   "f32" → types::Scalar(types::PrimitiveType::Float)
   ```

2. **Tensor Types**: Converts to flat pointer with shape information
   ```cpp
   "f32" + [32, 64] → types::Pointer(types::Scalar(types::PrimitiveType::Float))
   ```

3. **Shape**: Converts integer dimensions to symbolic expressions
   ```cpp
   [32, 64] → [symbolic::integer(32), symbolic::integer(64)]
   ```

### Operation Mapping Flow

1. **Query**: Given an operation name (e.g., "add")
2. **Lookup**: Map to SDFG library node code (e.g., "Add")
3. **Classification**: Determine if unary, binary, or reduce operation
4. **Use**: Create appropriate library node with the mapped code

## Usage Example (C++)

```cpp
#include "sdfg/frontend/mlir_frontend.h"

sdfg::frontend::MLIRFrontend frontend;

// Convert scalar type
auto scalar = frontend.convert_scalar_type("f32");
// scalar is types::Scalar(types::PrimitiveType::Float)

// Convert tensor type
std::vector<int64_t> shape = {32, 64};
auto tensor_ptr = frontend.convert_tensor_type("f32", shape);
// tensor_ptr is types::Pointer to types::Scalar(Float)

// Get symbolic shape
auto symbolic_shape = frontend.shape_to_symbolic(shape);
// Use with library nodes

// Map operations
auto add_code = frontend.get_elementwise_op_code("add");
// add_code is "Add" - can be used to create AddNode

auto sum_code = frontend.get_reduce_op_code("sum");
// sum_code is "Sum" - can be used to create SumNode
```

## Usage Example (Python)

```python
from sdfglib.mlir_frontend import (
    convert_scalar_type,
    convert_tensor_type,
    get_elementwise_op_code,
    get_reduce_op_code
)

# Convert types
scalar_type = convert_scalar_type("f32")  # SDFGPrimitiveType.FLOAT

element_type, shape = convert_tensor_type("f32", [32, 64])
# element_type: SDFGPrimitiveType.FLOAT
# shape: [32, 64]

# Map operations
add_code = get_elementwise_op_code("add")  # "Add"
sum_code = get_reduce_op_code("sum")  # "Sum"
```

## Key Design Decisions

1. **Flat Pointers for Tensors**: Aligns with SDFG's existing tensor library nodes that expect flat pointers with linearized indexing.

2. **No Full MLIR Dialect**: Instead of implementing a complete MLIR dialect (which would require MLIR infrastructure, TableGen, etc.), we created a lightweight C++ interface that provides the essential type conversions and operation mappings.

3. **Mapping to Existing Library Nodes**: Rather than creating new node types, we map MLIR operations to existing well-tested library nodes in sdfglib.

4. **Python Parity**: Both C++ and Python frontends provide the same functionality for flexibility in usage.

5. **Comprehensive Testing**: Both C++ and Python implementations are fully tested.

## Files Added

```
include/sdfg/frontend/
  └── mlir_frontend.h          # C++ header
src/frontend/
  └── mlir_frontend.cpp        # C++ implementation
tests/frontend/
  └── mlir_frontend_test.cpp   # C++ tests
python/sdfglib/
  ├── __init__.py              # Python package init
  └── mlir_frontend.py         # Python implementation
python/tests/
  └── test_mlir_frontend.py    # Python tests
```

## Test Results

- **C++ Tests**: 7/7 passing (100%)
- **Python Tests**: 18/18 passing (100%)
- **All Repository Tests**: 900/900 passing (100%)
- **Code Formatting**: All files formatted with clang-format and black

## Future Extensions

This frontend provides a foundation for:
1. Building full torch-mlir to SDFG conversion pipelines
2. Adding more operation types (convolution, pooling, etc.)
3. Implementing MLIR passes for optimization
4. Creating higher-level abstractions for common ML patterns
