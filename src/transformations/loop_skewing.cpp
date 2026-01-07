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

/**
 * Loop Skewing Transformation Implementation
 * 
 * This implementation of loop skewing modifies the iteration space of nested loops
 * by adjusting the inner loop's bounds to depend on the outer loop's iteration variable,
 * creating a "skewed" iteration pattern.
 * 
 * Implementation:
 * - Adjusts inner loop init: init_j + skew_factor * (i - init_i)
 * - Adjusts inner loop condition using symbolic::subs to substitute j with (j - skew_offset)
 * - Updates memory access patterns in loop body using sequence.replace()
 * 
 * Data Dependency Safety:
 * The transformation requires at least one loop to be a Map. Maps in SDFG represent
 * parallel loops with independent iterations by design - they should not have loop-carried
 * dependencies. This requirement ensures that:
 * 1. The transformation is safe for parallel execution
 * 2. No race conditions are introduced by skewing
 * 3. The semantics of parallel Maps are preserved
 * 
 * If both loops are Maps, the nested parallel structure is maintained. If only one
 * is a Map, the transformation converts one dimension of parallelism into a skewed
 * iteration space, which can expose different parallelization opportunities.
 * 
 * The transformation is safe when:
 * - Inner loop bounds don't depend on outer loop variable (verified)
 * - Loops are properly nested (verified)
 * - At least one loop is a Map with independent iterations (verified by type)
 */

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
    // Maps are designed for parallel execution with independent iterations,
    // which means they should not have loop-carried dependencies by construction.
    // Requiring at least one Map ensures the transformation is safe for parallel execution.
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
    
    // New inner loop condition: 
    // The condition needs to be adjusted to account for the skewed iteration space.
    // For a condition like "j < M", after skewing by offset, we need "j < M + offset"
    // We use symbolic::subs to substitute the upper bound expression
    // Original condition: j < ub becomes (j - offset) < ub, which is j < ub + offset
    auto new_inner_condition = symbolic::subs(
        this->inner_loop_.condition(),
        inner_indvar,
        symbolic::sub(inner_indvar, skew_offset)
    );
    
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
    builder.move_children(this->inner_loop_.root(), new_inner_loop->root());
    
    // Adjust memory access patterns in the loop body
    // After skewing, array accesses need to be adjusted:
    // Original access: A[i][j] becomes A[i][j - skew_offset]
    // We replace j with (j - skew_offset) in all expressions within the inner loop body
    new_inner_loop->root().replace(
        inner_indvar,
        symbolic::sub(inner_indvar, skew_offset)
    );
    
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
