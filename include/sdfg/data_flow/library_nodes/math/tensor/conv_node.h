/**
 * @file conv_node.h
 * @brief Convolution operation node compatible with ONNX Conv operator
 *
 * This file defines the ConvNode class which implements a tensor convolution
 * operation following the ONNX Conv operator specification. The node is expanded
 * using the im2col transformation into a GEMM operation.
 *
 * ## ONNX Conv Operator Compatibility
 *
 * The ConvNode implements the ONNX Conv operator with the following parameters:
 * - Input tensor X: [N, C_in, D1, D2, ..., Dn] for n-dimensional convolution
 * - Weight tensor W: [C_out, C_in/group, k1, k2, ..., kn]
 * - Optional bias tensor B: [C_out]
 * - Output tensor Y: [N, C_out, D1_out, D2_out, ..., Dn_out]
 *
 * Supported attributes:
 * - kernel_shape: Shape of the convolution kernel
 * - strides: Stride along each spatial axis
 * - pads: Padding for the beginning and ending along each spatial axis
 * - dilations: Dilation along each spatial axis
 * - group: Number of groups for grouped convolutions
 *
 * ## Expansion via im2col
 *
 * The convolution is expanded into a matrix multiplication using the im2col
 * (image to column) transformation:
 * 1. Transform input patches into columns
 * 2. Perform matrix multiplication (GEMM)
 * 3. Add bias (if present)
 * 4. Reshape to output tensor
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Conv("ml::Conv");

/**
 * @class ConvNode
 * @brief Convolution operation following ONNX Conv operator specification
 *
 * ConvNode represents a convolution operation that is compatible with the
 * ONNX Conv operator. The operation is expanded using the im2col transformation
 * into a GEMM operation for efficient computation.
 *
 * ## Input/Output Requirements
 * - Input connector "X": Input tensor [N, C_in, D1, ..., Dn]
 * - Input connector "W": Weight tensor [C_out, C_in/group, k1, ..., kn]
 * - Input connector "B" (optional): Bias tensor [C_out]
 * - Output connector "Y": Output tensor [N, C_out, D1_out, ..., Dn_out]
 *
 * ## Example
 *
 * Creating a 2D convolution:
 * @code
 * std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
 * std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
 * std::vector<symbolic::Expression> pads = {symbolic::integer(1), symbolic::integer(1),
 *                                            symbolic::integer(1), symbolic::integer(1)};
 * std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
 * auto group = symbolic::integer(1);
 *
 * auto& conv_node = builder.add_library_node<math::tensor::ConvNode>(
 *     block, debug_info, kernel_shape, strides, pads, dilations, group
 * );
 * @endcode
 */
class ConvNode : public TensorNode {
protected:
    std::vector<symbolic::Expression> kernel_shape_; ///< Shape of convolution kernel
    std::vector<symbolic::Expression> strides_;      ///< Stride along each spatial axis
    std::vector<symbolic::Expression> pads_;         ///< Padding (start and end for each axis)
    std::vector<symbolic::Expression> dilations_;    ///< Dilation along each spatial axis
    symbolic::Expression group_;                      ///< Number of groups for grouped convolution

public:
    /**
     * @brief Construct a convolution node
     * @param element_id Unique element identifier
     * @param debug_info Debug information
     * @param vertex Graph vertex
     * @param parent Parent dataflow graph
     * @param kernel_shape Shape of the convolution kernel
     * @param strides Stride along each spatial axis (defaults to 1 for each axis)
     * @param pads Padding for start and end of each axis (defaults to 0)
     * @param dilations Dilation along each spatial axis (defaults to 1)
     * @param group Number of groups for grouped convolution (defaults to 1)
     */
    ConvNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& kernel_shape,
        const std::vector<symbolic::Expression>& strides,
        const std::vector<symbolic::Expression>& pads,
        const std::vector<symbolic::Expression>& dilations,
        symbolic::Expression group
    );

    /**
     * @brief Get the convolution kernel shape
     * @return Kernel shape vector
     */
    const std::vector<symbolic::Expression>& kernel_shape() const { return kernel_shape_; }

    /**
     * @brief Get the stride values
     * @return Stride vector
     */
    const std::vector<symbolic::Expression>& strides() const { return strides_; }

    /**
     * @brief Get the padding values
     * @return Padding vector (start and end for each axis)
     */
    const std::vector<symbolic::Expression>& pads() const { return pads_; }

    /**
     * @brief Get the dilation values
     * @return Dilation vector
     */
    const std::vector<symbolic::Expression>& dilations() const { return dilations_; }

    /**
     * @brief Get the group count
     * @return Number of groups for grouped convolution
     */
    symbolic::Expression group() const { return group_; }

    void validate(const Function& function) const override;

    /**
     * @brief Expand convolution into im2col transformation + GEMM
     *
     * Expands the convolution operation by:
     * 1. Creating im2col transformation to convert input patches to columns
     * 2. Reshaping weights appropriately
     * 3. Creating GEMM node for matrix multiplication
     * 4. Adding bias if present
     * 5. Reshaping output to final tensor shape
     *
     * @param builder SDFG builder
     * @param analysis_manager Analysis manager
     * @return True if expansion succeeded
     */
    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    bool supports_integer_types() const override { return false; }

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;
};

/**
 * @class ConvNodeSerializer
 * @brief Serializer for ConvNode
 */
class ConvNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
