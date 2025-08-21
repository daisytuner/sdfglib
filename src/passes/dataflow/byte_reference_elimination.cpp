#include "sdfg/passes/dataflow/byte_reference_elimination.h"

#include "sdfg/analysis/users.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace passes {

ByteReferenceElimination::ByteReferenceElimination()
    : Pass() {

      };

std::string ByteReferenceElimination::name() { return "ByteReferenceElimination"; };

bool ByteReferenceElimination::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();

    for (auto& name : sdfg.containers()) {
        if (!sdfg.is_transient(name)) {
            continue;
        }
        if (!dynamic_cast<const types::Pointer*>(&sdfg.type(name))) {
            continue;
        }
        auto views = users.views(name);
        auto moves = users.moves(name);
        if (moves.size() != 1) {
            continue;
        }
        auto move = *moves.begin();
        auto move_node = dynamic_cast<data_flow::AccessNode*>(move->element());
        auto& move_graph = move_node->get_parent();
        auto& move_edge = *move_graph.in_edges(*move_node).begin();
        auto& move_type = move_edge.base_type();
        if (move_type.type_id() != types::TypeID::Pointer) {
            continue;
        }
        auto& move_pointer_type = static_cast<const types::Pointer&>(move_type);
        auto& move_pointee_type = move_pointer_type.pointee_type();
        if (move_pointee_type.type_id() == types::TypeID::Scalar) {
            if (move_pointee_type.primitive_type() == types::PrimitiveType::Int8) {
                continue;
            }
        }
        auto move_pointee_type_bytes = types::get_type_size(move_pointee_type, false);
        if (!SymEngine::is_a<SymEngine::Integer>(*move_pointee_type_bytes)) {
            continue;
        }
        auto move_pointee_type_bytes_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(move_pointee_type_bytes
        );

        for (auto& view : views) {
            auto view_node = dynamic_cast<data_flow::AccessNode*>(view->element());
            if (!users.dominates(*move, *view)) {
                continue;
            }

            auto& view_graph = view_node->get_parent();
            auto& view_edge = *view_graph.out_edges(*view_node).begin();
            auto& view_type = view_edge.base_type();
            if (view_type.type_id() != types::TypeID::Pointer) {
                continue;
            }

            // Criterion: View must do address calculation in bytes
            auto& view_pointer_type = static_cast<const types::Pointer&>(view_type);
            auto& view_pointee_type = view_pointer_type.pointee_type();
            if (view_pointee_type.type_id() != types::TypeID::Scalar) {
                continue;
            }
            if (view_pointee_type.primitive_type() != types::PrimitiveType::Int8) {
                continue;
            }
            if (view_edge.subset().size() != 1) {
                continue;
            }
            auto offset = view_edge.subset().at(0);
            if (!SymEngine::is_a<SymEngine::Integer>(*offset)) {
                continue;
            }
            auto offset_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(offset);
            if (offset_int->as_int() % move_pointee_type_bytes_int->as_int() != 0) {
                continue;
            }

            // Convert bytes into elements
            auto offset_elements = symbolic::div(offset_int, move_pointee_type_bytes);
            view_edge.set_subset({offset_elements});
            view_edge.set_base_type(move_pointer_type);

            applied = true;
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
