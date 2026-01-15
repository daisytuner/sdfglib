#include "sdfg/passes/dataflow/trivial_reference_conversion.h"

#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/types/type.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {


TrivialReferenceConversion::
    TrivialReferenceConversion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {};

bool TrivialReferenceConversion::accept(structured_control_flow::Block& block) {
    bool applied = false;
    auto& dfg = block.dataflow();

    // Narrow type of reference memlets to int8
    for (auto& edge : dfg.edges()) {
        // Find memlets that are references
        if (edge.type() != data_flow::MemletType::Reference) {
            continue;
        }
        if (edge.subset().size() != 1) {
            continue;
        }
        if (!symbolic::eq(edge.subset().at(0), symbolic::zero())) {
            continue;
        }
        auto& base_type = edge.base_type();
        if (base_type.type_id() != types::TypeID::Pointer) {
            continue;
        }
        auto& base_type_ptr = static_cast<const types::Pointer&>(base_type);
        if (!base_type_ptr.has_pointee_type()) {
            continue;
        }
        auto& pointee_type = base_type_ptr.pointee_type();
        if (pointee_type.type_id() != types::TypeID::Pointer) {
            continue;
        }
        edge.set_base_type(types::Pointer(types::Scalar(types::PrimitiveType::Int8)));
        applied = true;
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
