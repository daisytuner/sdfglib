/**
 * @file mlir_frontend.cpp
 * @brief Implementation of MLIR-style frontend for SDFG type conversions
 */

#include "sdfg/frontend/mlir_frontend.h"

#include <sstream>
#include <stdexcept>

namespace sdfg {
namespace frontend {

// MLIR type string to SDFG PrimitiveType mapping
const std::unordered_map<std::string, types::PrimitiveType> MLIRFrontend::mlir_type_map_ = {
    {"i1", types::PrimitiveType::Bool},
    {"i8", types::PrimitiveType::Int8},
    {"i16", types::PrimitiveType::Int16},
    {"i32", types::PrimitiveType::Int32},
    {"i64", types::PrimitiveType::Int64},
    {"f16", types::PrimitiveType::Half},
    {"f32", types::PrimitiveType::Float},
    {"f64", types::PrimitiveType::Double},
    {"index", types::PrimitiveType::Int64}, // MLIR index type maps to i64
};

// MLIR elementwise operation name to SDFG library node code mapping
const std::unordered_map<std::string, std::string> MLIRFrontend::elementwise_op_map_ = {
    // Binary operations
    {"add", "Add"},
    {"sub", "Sub"},
    {"mul", "Mul"},
    {"div", "Div"},
    {"pow", "Pow"},
    {"minimum", "Minimum"},
    {"maximum", "Maximum"},

    // Unary operations
    {"abs", "Abs"},
    {"sqrt", "Sqrt"},
    {"exp", "Exp"},
    {"erf", "Erf"},
    {"sigmoid", "Sigmoid"},
    {"tanh", "Tanh"},
    {"relu", "Relu"},
    {"leaky_relu", "LeakyRelu"},
    {"elu", "Elu"},
    {"hard_sigmoid", "HardSigmoid"},
    {"cast", "Cast"},
};

// MLIR reduction operation name to SDFG library node code mapping
const std::unordered_map<std::string, std::string> MLIRFrontend::reduce_op_map_ = {
    {"sum", "Sum"},
    {"mean", "Mean"},
    {"std", "Std"},
    {"max", "Max"},
    {"min", "Min"},
    {"softmax", "Softmax"},
};

types::Scalar MLIRFrontend::convert_scalar_type(const std::string& mlir_type_str) {
    auto it = mlir_type_map_.find(mlir_type_str);
    if (it == mlir_type_map_.end()) {
        std::ostringstream oss;
        oss << "Unsupported MLIR scalar type: " << mlir_type_str;
        throw std::invalid_argument(oss.str());
    }

    return types::Scalar(it->second);
}

types::Pointer MLIRFrontend::convert_tensor_type(const std::string& element_type_str, const std::vector<int64_t>& shape) {
    // Convert element type to SDFG scalar
    types::Scalar element_type = convert_scalar_type(element_type_str);

    // Create flat pointer to scalar (tensor is represented as 1D array)
    // Shape information is used separately by library nodes
    return types::Pointer(element_type);
}

std::vector<symbolic::Expression> MLIRFrontend::shape_to_symbolic(const std::vector<int64_t>& shape) {
    std::vector<symbolic::Expression> symbolic_shape;
    symbolic_shape.reserve(shape.size());

    for (int64_t dim : shape) {
        symbolic_shape.push_back(symbolic::integer(dim));
    }

    return symbolic_shape;
}

std::string MLIRFrontend::get_elementwise_op_code(const std::string& op_name) {
    auto it = elementwise_op_map_.find(op_name);
    if (it == elementwise_op_map_.end()) {
        std::ostringstream oss;
        oss << "Unsupported elementwise operation: " << op_name;
        throw std::invalid_argument(oss.str());
    }

    return it->second;
}

std::string MLIRFrontend::get_reduce_op_code(const std::string& op_name) {
    auto it = reduce_op_map_.find(op_name);
    if (it == reduce_op_map_.end()) {
        std::ostringstream oss;
        oss << "Unsupported reduce operation: " << op_name;
        throw std::invalid_argument(oss.str());
    }

    return it->second;
}

bool MLIRFrontend::is_elementwise_unary(const std::string& op_name) {
    static const std::vector<std::string> unary_ops = {
        "abs", "sqrt", "exp", "erf", "sigmoid", "tanh", "relu", "leaky_relu", "elu", "hard_sigmoid", "cast"
    };

    return std::find(unary_ops.begin(), unary_ops.end(), op_name) != unary_ops.end();
}

bool MLIRFrontend::is_elementwise_binary(const std::string& op_name) {
    static const std::vector<std::string> binary_ops = {"add", "sub", "mul", "div", "pow", "minimum", "maximum"};

    return std::find(binary_ops.begin(), binary_ops.end(), op_name) != binary_ops.end();
}

bool MLIRFrontend::is_reduce_op(const std::string& op_name) {
    return reduce_op_map_.find(op_name) != reduce_op_map_.end();
}

} // namespace frontend
} // namespace sdfg
