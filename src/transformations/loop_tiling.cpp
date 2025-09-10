#include "sdfg/transformations/loop_tiling.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace transformations {

LoopTiling::LoopTiling(structured_control_flow::StructuredLoop& loop, size_t tile_size)
    : loop_(loop), tile_size_(tile_size) {};

std::string LoopTiling::name() const { return "LoopTiling"; };

bool LoopTiling::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (this->tile_size_ <= 1) {
        return false;
    }
    // Criterion contiguous loop
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    return analysis::LoopAnalysis::is_contiguous(&loop_, assumptions_analysis);
};

void LoopTiling::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&loop_));

    auto indvar = loop_.indvar();

    // Step 1: Define new outer loop
    auto outer_indvar_str = builder.find_new_name(indvar->get_name() + "_tile");
    builder.add_container(outer_indvar_str, sdfg.type(loop_.indvar()->get_name()));

    auto outer_indvar = symbolic::symbol(outer_indvar_str);
    auto outer_condition = symbolic::subs(loop_.condition(), indvar, outer_indvar);
    auto outer_update = symbolic::add(outer_indvar, symbolic::integer(this->tile_size_));

    structured_control_flow::StructuredLoop* outer_loop = nullptr;
    if (auto map = dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        outer_loop =
            &builder
                 .add_map_before(
                     *parent, loop_, outer_indvar, outer_condition, loop_.init(), outer_update, map->schedule_type()
                 )
                 .first;
    } else {
        outer_loop =
            &builder.add_for_before(*parent, loop_, outer_indvar, outer_condition, loop_.init(), outer_update).first;
    }

    // Step 2: Redefine inner loop
    auto inner_indvar = indvar;
    auto inner_init = outer_indvar;
    auto inner_condition_tile =
        symbolic::Lt(inner_indvar, symbolic::add(outer_indvar, symbolic::integer(this->tile_size_)));

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto old_bound = analysis::LoopAnalysis::canonical_bound(&loop_, assumptions_analysis);

    symbolic::Condition inner_condition = inner_condition_tile;
    if (old_bound == SymEngine::null) {
        inner_condition = symbolic::And(inner_condition_tile, loop_.condition());
    } else if (SymEngine::is_a<SymEngine::Integer>(*old_bound)) {
        size_t old_bound_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(old_bound)->as_uint();
        if ((old_bound_int % this->tile_size_) == 0) {
            inner_condition = inner_condition_tile;
        } else {
            inner_condition = symbolic::And(inner_condition_tile, loop_.condition());
        }
    } else {
        inner_condition = symbolic::And(inner_condition_tile, loop_.condition());
    }
    auto inner_update = symbolic::add(inner_indvar, symbolic::integer(1));
    loop_.update() = inner_update;
    loop_.condition() = inner_condition;
    loop_.init() = inner_init;

    // Step 3: Move inner loop body to outer loop body
    builder.insert(loop_, *parent, outer_loop->root(), builder.debug_info().get_region(loop_.debug_info().indices()));

    analysis_manager.invalidate_all();
};

void LoopTiling::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw std::runtime_error("Unsupported loop type for serialization of loop: " + loop_.indvar()->get_name());
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["tile_size"] = tile_size_;
};

LoopTiling LoopTiling::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    size_t tile_size = desc["tile_size"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);

    return LoopTiling(*loop, tile_size);
};

} // namespace transformations
} // namespace sdfg
