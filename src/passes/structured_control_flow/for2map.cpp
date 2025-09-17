#include "sdfg/passes/structured_control_flow/for2map.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/passes/pipeline.h"

namespace sdfg {
namespace passes {

For2Map::For2Map(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {

      };

bool For2Map::can_be_applied(structured_control_flow::For& for_stmt, analysis::AnalysisManager& analysis_manager) {
    // Criterion: Loop must not have dereference memlets
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView users_view(users, for_stmt);
    for (auto& move : users_view.moves()) {
        auto element = move->element();
        if (auto node = dynamic_cast<data_flow::AccessNode*>(element)) {
            auto& parent = node->get_parent();
            for (auto& iedge : parent.in_edges(*node)) {
                if (iedge.type() == data_flow::MemletType::Dereference_Src) {
                    return false;
                }
            }
        }
    }
    for (auto& view : users_view.views()) {
        auto element = view->element();
        if (auto node = dynamic_cast<data_flow::AccessNode*>(element)) {
            auto& parent = node->get_parent();
            for (auto& iedge : parent.out_edges(*node)) {
                if (iedge.type() == data_flow::MemletType::Dereference_Dst) {
                    return false;
                }
            }
        }
    }

    // Criterion: Loop must not have side-effecting body
    std::list<const structured_control_flow::ControlFlowNode*> queue = {&for_stmt.root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        if (auto block = dynamic_cast<const structured_control_flow::Block*>(current)) {
            for (auto& node : block->dataflow().nodes()) {
                if (auto library_node = dynamic_cast<const data_flow::LibraryNode*>(&node)) {
                    if (library_node->side_effect()) {
                        return false;
                    }
                }
            }
        } else if (auto seq = dynamic_cast<const structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < seq->size(); i++) {
                auto& child = seq->at(i).first;
                queue.push_back(&child);
            }
        } else if (auto ifelse = dynamic_cast<const structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < ifelse->size(); i++) {
                auto& branch = ifelse->at(i).first;
                queue.push_back(&branch);
            }
        } else if (auto loop = dynamic_cast<const structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&loop->root());
        } else if (auto while_stmt = dynamic_cast<const structured_control_flow::While*>(current)) {
            queue.push_back(&while_stmt->root());
        } else if (auto for_stmt = dynamic_cast<const structured_control_flow::Break*>(current)) {
            // Do nothing
        } else if (auto for_stmt = dynamic_cast<const structured_control_flow::Continue*>(current)) {
            // Do nothing
        } else if (auto for_stmt = dynamic_cast<const structured_control_flow::Return*>(current)) {
            return false;
        } else {
            throw InvalidSDFGException("Unknown control flow node type in For2Map pass.");
        }
    }

    // Criterion: loop must be data-parallel w.r.t containers
    auto& data_dependency_analysis = analysis_manager.get<analysis::DataDependencyAnalysis>();
    auto dependencies = data_dependency_analysis.dependencies(for_stmt);

    // a. No true dependencies (RAW) between iterations
    for (auto& dep : dependencies) {
        if (dep.second == analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE) {
            return false;
        }
    }

    // b. False dependencies (WAW) are limited to loop-local variables
    auto locals = users.locals(for_stmt.root());
    for (auto& dep : dependencies) {
        auto& container = dep.first;
        if (locals.find(container) == locals.end()) {
            return false;
        }
    }

    // c. indvar not used after for
    if (locals.find(for_stmt.indvar()->get_name()) != locals.end()) {
        return false;
    }

    return true;
}

void For2Map::apply(
    structured_control_flow::For& for_stmt,
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager
) {
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&for_stmt));

    // convert for to map
    builder.convert_for(*parent, for_stmt);
}

bool For2Map::accept(structured_control_flow::For& node) {
    if (!this->can_be_applied(node, analysis_manager_)) {
        return false;
    }

    this->apply(node, builder_, analysis_manager_);
    return true;
}

} // namespace passes
} // namespace sdfg
