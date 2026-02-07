#pragma once

#include <string>

#include "sdfg/data_flow/library_node.h"

namespace sdfg {
namespace onnx {

namespace blas {

inline data_flow::ImplementationType ImplementationType_ONNX{"ONNX"};

} // namespace blas

namespace tensor {

/**
 * @brief ONNX implementation type for tensor operations
 * Emits ONNX graph representations for tensor operations
 */
inline data_flow::ImplementationType ImplementationType_ONNX{"ONNX"};

} // namespace tensor

} // namespace onnx
} // namespace sdfg
