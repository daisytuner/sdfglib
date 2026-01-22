#include "sdfg/passes/offloading/blas2cublas_pass.h"

#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/targets/cuda/cuda.h"

namespace sdfg {
namespace tenstorrent {

Blas2CuBlas::Blas2CuBlas(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool Blas2CuBlas::accept(structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();
    for (auto& library_node : dataflow.nodes()) {
        if (auto lib_node = dynamic_cast<math::blas::BLASNode*>(&library_node)) {
            auto implType = try_library_node_implementation(lib_node->code(), lib_node->scalar_primitive());

            if (implType) {
                lib_node->implementation_type() = implType.value();
            }
        }
    }
    return false;
}

std::optional<data_flow::ImplementationType>
try_library_node_implementation(const data_flow::LibraryNodeCode& code, types::PrimitiveType data_type) {
    if (data_type == types::PrimitiveType::Float || data_type == types::PrimitiveType::Double) {
        if (code == math::blas::LibraryNodeType_GEMM.value()) {
            return cuda::blas::ImplementationType_CUBLASWithTransfers;
        } else if (code == math::blas::LibraryNodeType_DOT.value()) {
            return cuda::blas::ImplementationType_CUBLASWithTransfers;
        } else {
            return std::nullopt;
        }
    } else {
        return std::nullopt;
    }
}

} // namespace tenstorrent
} // namespace sdfg
