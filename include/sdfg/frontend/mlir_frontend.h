/**
 * @file mlir_frontend.h
 * @brief MLIR-style frontend for converting torch-mlir types to SDFG types
 *
 * This file provides utilities for converting torch-mlir scalar types into SDFG scalars
 * and tensor types into flat pointers, enabling integration with SDFG's tensor library nodes.
 *
 * ## Type Conversions
 *
 * The MLIR frontend provides type conversion functions:
 * - Scalar types (i32, f32, etc.) → sdfg::types::Scalar
 * - Tensor types (tensor<NxMxf32>) → sdfg::types::Pointer to scalar with shape information
 *
 * ## Layer Mapping
 *
 * Common ML layers are mapped to existing library nodes:
 * - Elementwise operations (add, mul, relu, etc.) → ElementWiseNode
 * - Reduction operations (sum, mean, max, etc.) → ReduceNode
 * - Matrix operations (matmul) → GemmNode
 *
 * ## Example
 *
 * Converting torch-mlir types to SDFG:
 * @code
 * frontend::MLIRFrontend frontend;
 *
 * // Convert scalar type
 * auto scalar_type = frontend.convert_scalar_type("f32");
 *
 * // Convert tensor type to flat pointer
 * std::vector<int64_t> shape = {32, 64};
 * auto tensor_ptr = frontend.convert_tensor_type("f32", shape);
 *
 * // Map an elementwise operation
 * auto& add_node = frontend.create_elementwise_op(
 *     builder, block, "add", shape, debug_info
 * );
 * @endcode
 *
 * @see types::Scalar for SDFG scalar types
 * @see types::Pointer for SDFG pointer types
 * @see math::tensor::ElementWiseNode for elementwise operations
 * @see math::tensor::ReduceNode for reduction operations
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace frontend {

/**
 * @class MLIRFrontend
 * @brief Frontend for converting MLIR-style types and operations to SDFG
 *
 * Provides utilities to convert torch-mlir scalar types to SDFG scalars,
 * tensor types to flat pointers, and map typical ML layer operations to
 * existing SDFG library nodes.
 */
class MLIRFrontend {
public:
    MLIRFrontend() = default;
    ~MLIRFrontend() = default;

    /**
     * @brief Convert MLIR scalar type string to SDFG scalar type
     *
     * Supported MLIR types:
     * - Integer types: i1, i8, i16, i32, i64
     * - Float types: f16, f32, f64
     * - Index type: index (maps to i64)
     *
     * @param mlir_type_str MLIR type string (e.g., "f32", "i64")
     * @return SDFG scalar type
     * @throws std::invalid_argument if type string is not supported
     */
    types::Scalar convert_scalar_type(const std::string& mlir_type_str);

    /**
     * @brief Convert MLIR tensor type to SDFG flat pointer type
     *
     * Converts tensor<NxMx...xtype> to a flat pointer of scalars.
     * The shape information is returned separately for use with library nodes.
     *
     * Example: tensor<32x64xf32> → Pointer(Scalar(Float))
     *
     * @param element_type_str Element type string (e.g., "f32")
     * @param shape Tensor shape dimensions
     * @return Flat pointer to scalar type
     * @throws std::invalid_argument if element type is not supported
     */
    types::Pointer convert_tensor_type(const std::string& element_type_str, const std::vector<int64_t>& shape);

    /**
     * @brief Convert shape to symbolic expressions
     *
     * @param shape Integer shape dimensions
     * @return Vector of symbolic expressions
     */
    static std::vector<symbolic::Expression> shape_to_symbolic(const std::vector<int64_t>& shape);

    /**
     * @brief Get library node code for elementwise operation
     *
     * Maps MLIR elementwise op names to SDFG library node codes.
     * Supported operations:
     * - Binary: add, sub, mul, div, pow, minimum, maximum
     * - Unary: abs, sqrt, exp, erf, sigmoid, tanh, relu, leaky_relu, elu
     *
     * @param op_name MLIR operation name
     * @return Library node code string
     * @throws std::invalid_argument if operation is not supported
     */
    static std::string get_elementwise_op_code(const std::string& op_name);

    /**
     * @brief Get library node code for reduction operation
     *
     * Maps MLIR reduction op names to SDFG library node codes.
     * Supported operations: sum, mean, std, max, min, softmax
     *
     * @param op_name MLIR operation name
     * @return Library node code string
     * @throws std::invalid_argument if operation is not supported
     */
    static std::string get_reduce_op_code(const std::string& op_name);

    /**
     * @brief Check if operation is elementwise unary
     *
     * @param op_name Operation name
     * @return True if operation is unary elementwise
     */
    static bool is_elementwise_unary(const std::string& op_name);

    /**
     * @brief Check if operation is elementwise binary
     *
     * @param op_name Operation name
     * @return True if operation is binary elementwise
     */
    static bool is_elementwise_binary(const std::string& op_name);

    /**
     * @brief Check if operation is a reduction
     *
     * @param op_name Operation name
     * @return True if operation is a reduction
     */
    static bool is_reduce_op(const std::string& op_name);

private:
    // Mapping tables for type and operation conversions
    static const std::unordered_map<std::string, types::PrimitiveType> mlir_type_map_;
    static const std::unordered_map<std::string, std::string> elementwise_op_map_;
    static const std::unordered_map<std::string, std::string> reduce_op_map_;
};

} // namespace frontend
} // namespace sdfg
