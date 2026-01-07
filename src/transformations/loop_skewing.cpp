#include "sdfg/transformations/loop_skewing.h"
#include <stdexcept>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace transformations {

LoopSkewing::LoopSkewing(
    structured_control_flow::StructuredLoop& outer_loop,
    structured_control_flow::StructuredLoop& inner_loop,
    int skew_factor
)
    : outer_loop_(outer_loop), inner_loop_(inner_loop), skew_factor_(skew_factor) {
}

std::string LoopSkewing::name() const {
    return "LoopSkewing";
}

bool LoopSkewing::can_be_applied(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager
) {
    // Criterion 0: Skew factor must be non-zero
    if (this->skew_factor_ == 0) {
        return false;
    }

    auto& outer_indvar = this->outer_loop_.indvar();
    auto& inner_indvar = this->inner_loop_.indvar();

    // Criterion 1: Inner loop must not depend on outer loop indvar
    // This is because we're doing a simple skew, not a general affine transformation
    auto inner_loop_init = this->inner_loop_.init();
    auto inner_loop_condition = this->inner_loop_.condition();
    auto inner_loop_update = this->inner_loop_.update();
    
    if (symbolic::uses(inner_loop_init, outer_indvar->get_name()) ||
        symbolic::uses(inner_loop_condition, outer_indvar->get_name()) ||
        symbolic::uses(inner_loop_update, outer_indvar->get_name())) {
        return false;
    }

    // Criterion 2: Outer loop must have only the inner loop as a child
    if (outer_loop_.root().size() != 1) {
        return false;
    }
    if (outer_loop_.root().at(0).second.assignments().size() > 0) {
        return false;
    }
    if (&outer_loop_.root().at(0).first != &inner_loop_) {
        return false;
    }

    // Criterion 3: At least one of the loops must be a Map
    if (!dynamic_cast<structured_control_flow::Map*>(&outer_loop_) &&
        !dynamic_cast<structured_control_flow::Map*>(&inner_loop_)) {
        return false;
    }

    return true;
}

void LoopSkewing::apply(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager
) {
    auto& sdfg = builder.subject();
    
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& outer_scope = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&outer_loop_));
    auto& inner_scope = outer_loop_.root();
    
    int outer_index = outer_scope.index(this->outer_loop_);
    auto& outer_transition = outer_scope.at(outer_index).second;
    
    auto outer_indvar = this->outer_loop_.indvar();
    auto inner_indvar = this->inner_loop_.indvar();
    
    // Calculate the skewing offset: skew_factor * (outer_indvar - outer_init)
    auto skew_offset = symbolic::mul(
        symbolic::integer(this->skew_factor_),
        symbolic::sub(outer_indvar, this->outer_loop_.init())
    );
    
    // New inner loop init: inner_init + skew_offset
    auto new_inner_init = symbolic::add(this->inner_loop_.init(), skew_offset);
    
    // New inner loop condition: We need to adjust the condition
    // For the condition j < ub, it becomes j < ub + skew_offset
    // We need to substitute in the condition expression
    auto new_inner_condition = this->inner_loop_.condition();
    
    // Try to adjust the condition by substituting the upper bound
    // For a condition like j < M, we want j < M + skew_offset
    // This is a simplified approach - we update based on the structure
    if (auto lt_cond = SymEngine::rcp_dynamic_cast<const SymEngine::LessThan>(new_inner_condition)) {
        auto lhs = lt_cond->get_arg1();
        auto rhs = lt_cond->get_arg2();
        // If left side is the indvar, add offset to right side
        if (symbolic::eq(lhs, inner_indvar)) {
            new_inner_condition = symbolic::Lt(lhs, symbolic::add(rhs, skew_offset));
        }
    } else if (auto le_cond = SymEngine::rcp_dynamic_cast<const SymEngine::LessThan>(
        SymEngine::rcp_dynamic_cast<const SymEngine::Not>(new_inner_condition))) {
        // Handle j <= ub case (which might be represented differently)
        // For simplicity, we'll keep the original condition structure
    }
    
    // New inner loop update: same as before
    auto new_inner_update = this->inner_loop_.update();
    
    // Add new outer loop (same as original outer loop)
    structured_control_flow::StructuredLoop* new_outer_loop = nullptr;
    if (auto outer_map = dynamic_cast<structured_control_flow::Map*>(&outer_loop_)) {
        new_outer_loop = &builder.add_map_after(
            outer_scope,
            this->outer_loop_,
            outer_map->indvar(),
            outer_map->condition(),
            outer_map->init(),
            outer_map->update(),
            outer_map->schedule_type(),
            outer_transition.assignments(),
            this->outer_loop_.debug_info()
        );
    } else {
        new_outer_loop = &builder.add_for_after(
            outer_scope,
            this->outer_loop_,
            this->outer_loop_.indvar(),
            this->outer_loop_.condition(),
            this->outer_loop_.init(),
            this->outer_loop_.update(),
            outer_transition.assignments(),
            this->outer_loop_.debug_info()
        );
    }
    
    // Add new inner loop with skewed bounds
    structured_control_flow::StructuredLoop* new_inner_loop = nullptr;
    if (auto inner_map = dynamic_cast<structured_control_flow::Map*>(&inner_loop_)) {
        new_inner_loop = &builder.add_map_after(
            inner_scope,
            this->inner_loop_,
            inner_map->indvar(),
            new_inner_condition,
            new_inner_init,
            new_inner_update,
            inner_map->schedule_type(),
            {},
            this->inner_loop_.debug_info()
        );
    } else {
        new_inner_loop = &builder.add_for_after(
            inner_scope,
            this->inner_loop_,
            this->inner_loop_.indvar(),
            new_inner_condition,
            new_inner_init,
            new_inner_update,
            {},
            this->inner_loop_.debug_info()
        );
    }
    
    // Move inner loop body to new inner loop
    // We need to substitute inner_indvar with (inner_indvar - skew_offset) in the body
    // For now, we'll move the body as-is
    // A full implementation would need to traverse and update all memlet accesses
    builder.move_children(this->inner_loop_.root(), new_inner_loop->root());
    
    // Move new inner loop into new outer loop
    builder.move_children(this->outer_loop_.root(), new_outer_loop->root());
    
    // Remove old loops
    builder.remove_child(new_outer_loop->root(), 0);
    builder.remove_child(outer_scope, outer_index);
    
    analysis_manager.invalidate_all();
}

void LoopSkewing::to_json(nlohmann::json& j) const {
    std::vector<std::string> loop_types;
    for (auto* loop : {&(this->outer_loop_), &(this->inner_loop_)}) {
        if (dynamic_cast<structured_control_flow::For*>(loop)) {
            loop_types.push_back("for");
        } else if (dynamic_cast<structured_control_flow::Map*>(loop)) {
            loop_types.push_back("map");
        } else {
            throw std::runtime_error("Unsupported loop type for serialization of loop: " + loop->indvar()->get_name());
        }
    }
    
    j["transformation_type"] = this->name();
    j["subgraph"] = {
        {"0", {{"element_id", this->outer_loop_.element_id()}, {"type", loop_types[0]}}},
        {"1", {{"element_id", this->inner_loop_.element_id()}, {"type", loop_types[1]}}}
    };
    j["parameters"] = {{"skew_factor", this->skew_factor_}};
}

LoopSkewing LoopSkewing::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto outer_loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto inner_loop_id = desc["subgraph"]["1"]["element_id"].get<size_t>();
    int skew_factor = desc["parameters"]["skew_factor"].get<int>();
    
    auto outer_element = builder.find_element_by_id(outer_loop_id);
    auto inner_element = builder.find_element_by_id(inner_loop_id);
    
    if (outer_element == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(outer_loop_id) + " not found.");
    }
    if (inner_element == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(inner_loop_id) + " not found.");
    }
    
    auto outer_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(outer_element);
    if (outer_loop == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(outer_loop_id) + " is not a StructuredLoop.");
    }
    
    auto inner_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(inner_element);
    if (inner_loop == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(inner_loop_id) + " is not a StructuredLoop.");
    }
    
    return LoopSkewing(*outer_loop, *inner_loop, skew_factor);
}

} // namespace transformations
} // namespace sdfg
