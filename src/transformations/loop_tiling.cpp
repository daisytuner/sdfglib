#include "sdfg/transformations/loop_tiling.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace transformations {

LoopTiling::LoopTiling(structured_control_flow::StructuredLoop& loop, size_t tile_size)
    : loop_(loop), tile_size_(tile_size) {};

std::string LoopTiling::name() const { return "LoopTiling"; };

bool LoopTiling::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) {
    if (this->tile_size_ <= 1) {
        return false;
    }
    // Criterion contiguous loop
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    return loop_analysis.is_contiguous(&loop_);
};

void LoopTiling::apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent =
        static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&loop_));

    auto indvar = loop_.indvar();
    auto outer_indvar_str = builder.find_new_name(indvar->get_name() + "_tile");
    builder.add_container(outer_indvar_str, sdfg.type(loop_.indvar()->get_name()));
    auto outer_indvar = symbolic::symbol(outer_indvar_str);

    auto tile_size = symbolic::integer(this->tile_size_);

    auto condition = symbolic::subs(loop_.condition(), indvar, outer_indvar);
    auto update = symbolic::add(outer_indvar, tile_size);

    auto& outer_loop =
        builder.add_for_before(*parent, loop_, outer_indvar, condition, loop_.init(), update).first;

    auto& outer_body = outer_loop.root();

    auto inner_indvar = indvar;
    auto inner_init = outer_indvar;
    auto inner_condition_tile = symbolic::Lt(inner_indvar, symbolic::add(outer_indvar, tile_size));
    auto inner_condition_base = symbolic::subs(loop_.condition(), outer_indvar, inner_indvar);
    auto inner_condition = symbolic::And(inner_condition_tile, inner_condition_base);
    auto inner_update = symbolic::add(inner_indvar, symbolic::integer(1));

    // Add new loop with original body
    auto& tmp_root = builder.add_sequence_before(*parent, loop_).first;
    auto& inner_loop =
        builder.add_for(tmp_root, inner_indvar, inner_condition, inner_init, inner_update);

    deepcopy::StructuredSDFGDeepCopy copies(builder, inner_loop.root(), loop_.root());
    copies.copy();

    builder.clear_sequence(outer_body);

    deepcopy::StructuredSDFGDeepCopy copies2(builder, outer_body, tmp_root);
    copies2.copy();

    builder.remove_child(*parent, tmp_root);
    builder.remove_child(*parent, loop_);

    analysis_manager.invalidate_all();

    passes::SequenceFusion sf_pass;
    passes::DeadCFGElimination dce_pass;
    bool applies = false;
    do {
        applies = false;
        applies |= dce_pass.run(builder, analysis_manager);
        applies |= sf_pass.run(builder, analysis_manager);
    } while (applies);
};

void LoopTiling::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["loop_element_id"] = loop_.element_id();
    j["tile_size"] = tile_size_;
};

LoopTiling LoopTiling::from_json(builder::StructuredSDFGBuilder& builder,
                                 const nlohmann::json& desc) {
    auto loop_id = desc["loop_element_id"].get<size_t>();
    size_t tile_size = desc["tile_size"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " +
                                                        std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::For*>(element);

    return LoopTiling(*loop, tile_size);
};

}  // namespace transformations
}  // namespace sdfg
