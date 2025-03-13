#include "sdfg/transformations/loop_tiling.h"

#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"

namespace sdfg {
namespace transformations {

LoopTiling::LoopTiling(structured_control_flow::Sequence& parent,
                       structured_control_flow::For& loop, size_t tile_size)
    : parent_(parent), loop_(loop), tile_size_(tile_size) {
    assert(tile_size > 1);
};

std::string LoopTiling::name() { return "LoopTiling"; };

bool LoopTiling::can_be_applied(Schedule& schedule) {
    // Criterion contiguous loop
    return analysis::DataParallelismAnalysis::is_contiguous(this->loop_);
};

void LoopTiling::apply(Schedule& schedule) {
    auto& builder = schedule.builder();
    auto& sdfg = builder.subject();

    auto indvar = loop_.indvar();
    auto outer_indvar_str = builder.find_new_name(indvar->get_name() + "_tile");
    builder.add_container(outer_indvar_str, sdfg.type(loop_.indvar()->get_name()));
    auto outer_indvar = symbolic::symbol(outer_indvar_str);

    auto tile_size = symbolic::integer(this->tile_size_);

    auto& outer_body = loop_.root();

    loop_.indvar() = outer_indvar;
    loop_.condition() = symbolic::subs(loop_.condition(), indvar, outer_indvar);
    loop_.update() = symbolic::subs(loop_.update(), indvar, outer_indvar);
    loop_.update() = symbolic::add(outer_indvar, tile_size);

    auto inner_indvar = indvar;
    auto inner_init = outer_indvar;
    auto inner_condition_tile = symbolic::Lt(inner_indvar, symbolic::add(outer_indvar, tile_size));
    auto inner_condition_base = symbolic::subs(loop_.condition(), outer_indvar, inner_indvar);
    auto inner_condition = symbolic::And(inner_condition_tile, inner_condition_base);
    auto inner_update = symbolic::add(inner_indvar, symbolic::integer(1));

    // Add new loop with original body
    auto& tmp_root = builder.add_sequence_before(parent_, loop_).first;
    auto& inner_loop =
        builder.add_for(tmp_root, inner_indvar, inner_condition, inner_init, inner_update);

    deepcopy::StructuredSDFGDeepCopy copies(builder, inner_loop.root(), outer_body);
    copies.copy();

    builder.clear_sequence(outer_body);

    deepcopy::StructuredSDFGDeepCopy copies2(builder, outer_body, tmp_root);
    copies2.copy();

    builder.remove_child(parent_, tmp_root);

    auto& analysis_manager = schedule.analysis_manager();
    analysis_manager.invalidate_all();

    passes::SequenceFusion sf_pass;
    passes::DeadCFGElimination dce_pass;
    bool applies = false;
    do {
        applies = false;
        applies |= dce_pass.run(schedule.builder(), analysis_manager);
        applies |= sf_pass.run(schedule.builder(), analysis_manager);
    } while (applies);
};

}  // namespace transformations
}  // namespace sdfg
