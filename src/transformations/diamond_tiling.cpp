#include "sdfg/transformations/diamond_tiling.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_tiling.h"

namespace sdfg {
namespace transformations {

DiamondTiling::DiamondTiling(
    structured_control_flow::StructuredLoop& outer_loop,
    structured_control_flow::StructuredLoop& inner_loop,
    size_t outer_tile_size,
    size_t inner_tile_size
)
    : outer_loop_(outer_loop),
      inner_loop_(inner_loop),
      outer_tile_size_(outer_tile_size),
      inner_tile_size_(inner_tile_size) {}

std::string DiamondTiling::name() const { return "DiamondTiling"; }

bool DiamondTiling::can_be_applied(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
) {
    // Check tile sizes are valid
    if (this->outer_tile_size_ <= 1 || this->inner_tile_size_ <= 1) {
        return false;
    }

    // Check that both loops are contiguous
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    if (!analysis::LoopAnalysis::is_contiguous(&outer_loop_, assumptions_analysis)) {
        return false;
    }
    if (!analysis::LoopAnalysis::is_contiguous(&inner_loop_, assumptions_analysis)) {
        return false;
    }

    // Check that inner loop is the only child of outer loop body
    if (outer_loop_.root().size() != 1) {
        return false;
    }
    if (outer_loop_.root().at(0).second.assignments().size() > 0) {
        return false;
    }
    if (&outer_loop_.root().at(0).first != &inner_loop_) {
        return false;
    }

    // Check that inner loop does not depend on outer loop induction variable
    auto& outer_indvar = this->outer_loop_.indvar();
    auto inner_loop_init = this->inner_loop_.init();
    auto inner_loop_condition = this->inner_loop_.condition();
    auto inner_loop_update = this->inner_loop_.update();
    if (symbolic::uses(inner_loop_init, outer_indvar->get_name()) ||
        symbolic::uses(inner_loop_condition, outer_indvar->get_name()) ||
        symbolic::uses(inner_loop_update, outer_indvar->get_name())) {
        return false;
    }

    return true;
}

void DiamondTiling::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Step 1: Apply tiling to the outer loop
    LoopTiling outer_tiling(outer_loop_, outer_tile_size_);
    outer_tiling.apply(builder, analysis_manager);

    // After outer tiling, we need to find the new loops:
    // - The tiled outer loop is at the top level (replaces the original outer_loop_)
    // - The original outer_loop_ (modified bounds) is now nested inside the tiled loop
    // - The inner_loop_ is still nested inside the original outer_loop_
    
    // The structure after outer tiling is:
    // Level 0: outer_loop_tile
    //   Level 1: outer_loop (with modified bounds)
    //     Level 2: inner_loop

    // Step 2: Apply tiling to the inner loop
    LoopTiling inner_tiling(inner_loop_, inner_tile_size_);
    inner_tiling.apply(builder, analysis_manager);

    // After inner tiling, the structure is:
    // Level 0: outer_loop_tile
    //   Level 1: outer_loop (modified)
    //     Level 2: inner_loop_tile
    //       Level 3: inner_loop (with modified bounds)

    // Step 3: Interchange the original outer_loop with the tiled inner_loop_tile
    // to create the diamond pattern
    
    // outer_loop is still valid and is at level 1
    // Find the inner_loop_tile which should be the first child of outer_loop's body
    auto& inner_loop_tile_ref = outer_loop_.root().at(0).first;
    auto* inner_loop_tile = dynamic_cast<structured_control_flow::StructuredLoop*>(&inner_loop_tile_ref);

    // Now interchange outer_loop with inner_loop_tile to create the diamond pattern
    LoopInterchange interchange(outer_loop_, *inner_loop_tile);
    interchange.apply(builder, analysis_manager);

    // Final structure after all transformations:
    // Level 0: outer_loop_tile
    //   Level 1: inner_loop_tile (swapped with outer_loop)
    //     Level 2: outer_loop (swapped with inner_loop_tile)
    //       Level 3: inner_loop

    analysis_manager.invalidate_all();
}

void DiamondTiling::to_json(nlohmann::json& j) const {
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
    j["parameters"] = {{"outer_tile_size", outer_tile_size_}, {"inner_tile_size", inner_tile_size_}};
}

DiamondTiling DiamondTiling::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto outer_loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto inner_loop_id = desc["subgraph"]["1"]["element_id"].get<size_t>();
    size_t outer_tile_size = desc["parameters"]["outer_tile_size"].get<size_t>();
    size_t inner_tile_size = desc["parameters"]["inner_tile_size"].get<size_t>();
    
    auto outer_element = builder.find_element_by_id(outer_loop_id);
    auto inner_element = builder.find_element_by_id(inner_loop_id);
    
    if (outer_element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(outer_loop_id) + " not found.");
    }
    if (inner_element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(inner_loop_id) + " not found.");
    }
    
    auto outer_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(outer_element);
    if (outer_loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(outer_loop_id) + " is not a StructuredLoop."
        );
    }
    
    auto inner_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(inner_element);
    if (inner_loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(inner_loop_id) + " is not a StructuredLoop."
        );
    }

    return DiamondTiling(*outer_loop, *inner_loop, outer_tile_size, inner_tile_size);
}

} // namespace transformations
} // namespace sdfg
