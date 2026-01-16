# MLIR to SDFG Conversion Library

This directory contains the MLIR to SDFG conversion infrastructure for importing models from torch-mlir and other MLIR dialects.

## Overview

This library provides lowering passes that convert MLIR operations to SDFG (Stateful DataFlow Graph) library nodes. It focuses on:

1. **Torch-MLIR Integration**: Converting torch-mlir operations to SDFG
2. **Linalg Operations**: Converting structured linear algebra operations to SDFG library nodes
3. **No IREE Dependency**: Unlike previous implementations, this avoids IREE dependencies

## CMake Best Practices

This project follows **canonical MLIR CMake patterns**. See [`CMAKE_BEST_PRACTICES.md`](./CMAKE_BEST_PRACTICES.md) for detailed explanation of:
- Why we add `MLIR_CMAKE_DIR` to `CMAKE_MODULE_PATH`
- Why we use `add_mlir_library()` instead of `add_library()`
- How to properly link against MLIR components
- Troubleshooting common CMake issues

## Structure

```
mlir/
├── CMakeLists.txt           # Build configuration (not built by default)
├── include/mlir-sdfg/       # Public headers
│   └── Conversion/
│       ├── Passes.h         # Pass declarations
│       ├── TorchToSDFG.h    # Torch dialect conversion
│       └── LinalgToSDFG.h   # Linalg dialect conversion
├── lib/Conversion/          # Implementation
│   ├── TorchToSDFG.cpp      # Torch conversion implementation
│   ├── LinalgToSDFG.cpp     # Linalg conversion implementation
│   └── PassRegistry.cpp     # Pass registration
└── tests/                   # Unit tests (when enabled)
```

## Building

This library is **not built by default**. To enable it:

```bash
cmake -DDOCC_BUILD_MLIR=ON ..
```

### Requirements

- LLVM/MLIR 19.x
- sdfglib (linked automatically)
- torch-mlir (for Torch dialect support)

## Conversion Passes

### TorchToSDFG Pass

Converts torch-mlir operations to SDFG library nodes:

- `torch.aten.matmul` → SDFG GEMM library node
- `torch.aten.conv2d` → SDFG Conv library node
- `torch.aten.relu` → SDFG activation node
- etc.

**Usage:**
```bash
mlir-opt --convert-torch-to-sdfg input.mlir
```

### LinalgToSDFG Pass

Converts Linalg structured operations to SDFG library nodes:

- `linalg.matmul` → SDFG GEMM library node
- `linalg.conv_2d_nhwc_hwcf` → SDFG Conv library node
- `linalg.pooling_*` → SDFG pooling nodes
- etc.

**Usage:**
```bash
mlir-opt --convert-linalg-to-sdfg input.mlir
```

## Integration with Python Frontend

The Python bindings in `python/docc/pytorch_integration.py` will link against this library to convert PyTorch models:

```python
# PyTorch model → torch.jit.trace → Torch-MLIR → SDFG
optimized = docc.import_pytorch_model(model, example_inputs)
```

Implementation path:
1. Use `torch.jit.trace()` to get TorchScript
2. Convert to torch-mlir using torch-mlir compiler
3. Apply `--convert-torch-to-sdfg` pass
4. Generate SDFG and compile for inference

## Library Node Mapping

The conversion focuses on mapping operations to SDFG library nodes:

| MLIR Operation | SDFG Library Node | Status |
|----------------|-------------------|--------|
| `linalg.matmul` | GEMM | TODO |
| `linalg.conv_2d` | Conv2D | TODO |
| `torch.aten.mm` | GEMM | TODO |
| `torch.aten.conv2d` | Conv2D | TODO |
| `torch.aten.relu` | Activation | TODO |
| `torch.aten.max_pool2d` | Pooling | TODO |

## Design Notes

### Focus on Library Nodes

Rather than lowering to fine-grained SDFG operations, we target SDFG library nodes (GEMM, Conv, etc.) which:
- Provide optimized implementations
- Are portable across targets
- Simplify the conversion logic

### Avoiding IREE

Previous implementations used IREE for MLIR execution. This approach:
- Avoids IREE dependency complexity
- Directly targets SDFG as the execution model
- Gives us full control over optimization pipeline

### Torch-MLIR Backend

We use torch-mlir's backend dialects which are:
- Well-defined and stable
- Already lowered from high-level PyTorch ops
- Easier to map to SDFG library nodes

## Future Work

1. **Implement conversion patterns** for common operations
2. **Add shape inference** and type conversion
3. **Support dynamic shapes** where possible
4. **Add comprehensive tests** for each conversion pattern
5. **Optimize patterns** for common operation combinations
6. **Add Python bindings** to expose passes to Python frontend

## References

- [torch-mlir](https://github.com/llvm/torch-mlir)
- [MLIR](https://mlir.llvm.org/)
- [SDFG Library](../sdfglib/)
