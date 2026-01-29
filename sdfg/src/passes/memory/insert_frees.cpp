#include "sdfg/passes/memory/insert_frees.h"

#include "sdfg/analysis/escape_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/types/pointer.h"

namespace sdfg {
namespace passes {

InsertFrees::InsertFrees(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

namespace {

bool is_loop(structured_control_flow::ControlFlowNode* node) {
    return dynamic_cast<structured_control_flow::StructuredLoop*>(node) != nullptr ||
           dynamic_cast<structured_control_flow::While*>(node) != nullptr;
}

structured_control_flow::ControlFlowNode*
find_containing_loop(structured_control_flow::Block* block, analysis::ScopeAnalysis& scope_analysis) {
    auto ancestors = scope_analysis.ancestor_scopes(block);
    for (auto* ancestor : ancestors) {
        if (is_loop(ancestor)) {
            return ancestor;
        }
    }
    return nullptr;
}

bool same_loop_context(
    structured_control_flow::Block* malloc_block,
    structured_control_flow::Block* use_block,
    analysis::ScopeAnalysis& scope_analysis
) {
    auto* malloc_loop = find_containing_loop(malloc_block, scope_analysis);
    auto* use_loop = find_containing_loop(use_block, scope_analysis);

    if (malloc_loop == nullptr && use_loop == nullptr) {
        // Neither is in a loop
        return true;
    }

    if (malloc_loop == use_loop) {
        // Both in the same innermost loop
        return true;
    }

    if (malloc_loop == nullptr && use_loop != nullptr) {
        // Malloc outside loop, use inside loop - different context
        return false;
    }

    if (malloc_loop != nullptr && use_loop != nullptr) {
        auto use_ancestors = scope_analysis.ancestor_scopes(use_block);
        for (auto* ancestor : use_ancestors) {
            if (ancestor == malloc_loop) {
                return false;
            }
        }
    }

    return false;
}

} // anonymous namespace

void InsertFrees::insert_free_after(const std::string& container, analysis::User* last_use) {
    auto& sdfg = builder_.subject();
    auto& escape_analysis = analysis_manager_.get<analysis::EscapeAnalysis>();
    auto& scope_analysis = analysis_manager_.get<analysis::ScopeAnalysis>();

    auto* element = last_use->element();
    if (!element) {
        return;
    }

    structured_control_flow::Block* use_block = nullptr;
    if (auto* access_node = dynamic_cast<data_flow::AccessNode*>(element)) {
        auto* parent = access_node->get_parent().get_parent();
        use_block = dynamic_cast<structured_control_flow::Block*>(parent);
    } else if (auto* lib_node = dynamic_cast<data_flow::LibraryNode*>(element)) {
        auto* parent = lib_node->get_parent().get_parent();
        use_block = dynamic_cast<structured_control_flow::Block*>(parent);
    }

    if (!use_block) {
        return;
    }

    auto* malloc_block = escape_analysis.malloc_block(container);
    if (!malloc_block) {
        return;
    }

    structured_control_flow::ControlFlowNode* insert_after_node = use_block;
    structured_control_flow::Sequence* parent_sequence = nullptr;

    if (same_loop_context(malloc_block, use_block, scope_analysis)) {
        auto* parent_scope = scope_analysis.parent_scope(use_block);
        parent_sequence = dynamic_cast<structured_control_flow::Sequence*>(parent_scope);
    } else {
        auto* malloc_loop = find_containing_loop(malloc_block, scope_analysis);
        auto use_ancestors = scope_analysis.ancestor_scopes(use_block);

        structured_control_flow::ControlFlowNode* target_loop = nullptr;
        for (auto it = use_ancestors.rbegin(); it != use_ancestors.rend(); ++it) {
            if (is_loop(*it)) {
                if (*it != malloc_loop) {
                    target_loop = *it;
                    break;
                }
            }
        }

        if (target_loop) {
            insert_after_node = target_loop;
            auto* loop_parent = scope_analysis.parent_scope(target_loop);
            parent_sequence = dynamic_cast<structured_control_flow::Sequence*>(loop_parent);
        } else {
            auto* parent_scope = scope_analysis.parent_scope(use_block);
            parent_sequence = dynamic_cast<structured_control_flow::Sequence*>(parent_scope);
        }
    }

    if (!parent_sequence) {
        return;
    }

    DebugInfo debug_info;
    auto& free_block = builder_.add_block_after(*parent_sequence, *insert_after_node, {}, debug_info);

    auto& free_node = builder_.add_library_node<stdlib::FreeNode>(free_block, debug_info);

    auto& input_access = builder_.add_access(free_block, container, debug_info);
    auto& output_access = builder_.add_access(free_block, container, debug_info);

    types::Pointer opaque_ptr;
    builder_
        .add_computational_memlet(free_block, input_access, free_node, "_ptr", data_flow::Subset{}, opaque_ptr, debug_info);

    builder_.add_computational_memlet(
        free_block, free_node, "_ptr", output_access, data_flow::Subset{}, opaque_ptr, debug_info
    );
}

bool InsertFrees::visit() {
    auto& escape_analysis = analysis_manager_.get<analysis::EscapeAnalysis>();

    bool applied = false;

    auto non_escaping = escape_analysis.non_escaping_allocations();

    for (const auto& container : non_escaping) {
        if (freed_containers_.count(container) > 0) {
            continue;
        }

        auto* last_use = escape_analysis.last_use(container);
        if (!last_use) {
            continue;
        }

        insert_free_after(container, last_use);
        freed_containers_.insert(container);
        applied = true;
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
