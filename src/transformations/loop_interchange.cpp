#include "sdfg/transformations/loop_interchange.h"

#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace transformations {

LoopInterchange::LoopInterchange(structured_control_flow::Sequence& parent,
                                 structured_control_flow::StructuredLoop& outer_loop,
                                 structured_control_flow::StructuredLoop& inner_loop)
    : outer_loop_(outer_loop), inner_loop_(inner_loop) {

      };

std::string LoopInterchange::name() { return "LoopInterchange"; };

bool LoopInterchange::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                     analysis::AnalysisManager& analysis_manager) {
    auto& outer_indvar = this->outer_loop_.indvar();

    // Criterion: Inner loop must not depend on outer loop
    auto& inner_loop_init = this->inner_loop_.init();
    auto& inner_loop_condition = this->inner_loop_.condition();
    auto& inner_loop_update = this->inner_loop_.update();
    if (symbolic::uses(inner_loop_init, outer_indvar->get_name()) ||
        symbolic::uses(inner_loop_condition, outer_indvar->get_name()) ||
        symbolic::uses(inner_loop_update, outer_indvar->get_name())) {
        return false;
    }

    // Criterion: Outer loop must not have any outer blocks
    if (outer_loop_.root().size() > 1) {
        return false;
    }
    if (outer_loop_.root().at(0).second.assignments().size() > 0) {
        return false;
    }
    if (&outer_loop_.root().at(0).first != &inner_loop_) {
        return false;
    }
    // Criterion: Any of both loops is a map
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();

    auto& outer_dependencies = analysis.get(this->outer_loop_);
    if (outer_dependencies.size() > 0) {
        bool is_map = true;
        for (auto& dep : outer_dependencies) {
            auto& dep_type = dep.second;
            if (dep_type < analysis::Parallelism::PARALLEL) {
                is_map = false;
                break;
            }
        }
        if (is_map) {
            return true;
        }
    }
    auto& inner_dependencies = analysis.get(this->inner_loop_);
    if (inner_dependencies.size() > 0) {
        bool is_map = true;
        for (auto& dep : inner_dependencies) {
            auto& dep_type = dep.second;
            if (dep_type < analysis::Parallelism::PARALLEL) {
                is_map = false;
                break;
            }
        }
        if (is_map) {
            return true;
        }
    }

    return false;
};

void LoopInterchange::apply(builder::StructuredSDFGBuilder& builder,
                            analysis::AnalysisManager& analysis_manager) {
    auto new_inner_loop = builder.add_for_after(
        builder.parent(inner_loop_), this->inner_loop_, this->outer_loop_.indvar(),
        this->outer_loop_.condition(), this->outer_loop_.init(), this->outer_loop_.update());
    auto& inner_body = this->inner_loop_.root();

    builder.insert_children(new_inner_loop.first.root(), inner_body, 0);

    auto new_outer_loop = builder.add_for_after(
        builder.parent(outer_loop_), this->outer_loop_, this->inner_loop_.indvar(),
        this->inner_loop_.condition(), this->inner_loop_.init(), this->inner_loop_.update());

    auto& outer_body = this->outer_loop_.root();
    builder.insert_children(new_outer_loop.first.root(), outer_body, 0);
    builder.remove_child(builder.parent(inner_loop_), this->inner_loop_);
    builder.remove_child(builder.parent(outer_loop_), this->outer_loop_);

    analysis_manager.invalidate_all();
};

}  // namespace transformations
}  // namespace sdfg
