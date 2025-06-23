#include "sdfg/transformations/loop_interchange.h"

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace transformations {

LoopInterchange::LoopInterchange(structured_control_flow::StructuredLoop& outer_loop,
                                 structured_control_flow::StructuredLoop& inner_loop)
    : outer_loop_(outer_loop), inner_loop_(inner_loop) {

      };

std::string LoopInterchange::name() const { return "LoopInterchange"; };

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
    if (dynamic_cast<structured_control_flow::Map*>(&outer_loop_) ||
        dynamic_cast<structured_control_flow::Map*>(&inner_loop_)) {
        return true;
    }

    return false;
};

void LoopInterchange::apply(builder::StructuredSDFGBuilder& builder,
                            analysis::AnalysisManager& analysis_manager) {
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& outer_scope =
        static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&outer_loop_));
    auto& inner_scope =
        static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&inner_loop_));

    // Add new outer loop behind current outer loop
    structured_control_flow::StructuredLoop* new_outer_loop = nullptr;
    if (auto inner_map = dynamic_cast<structured_control_flow::Map*>(&inner_loop_)) {
        new_outer_loop = &builder
                              .add_map_after(outer_scope, this->outer_loop_, inner_map->indvar(),
                                             inner_map->condition(), inner_map->init(),
                                             inner_map->update(), inner_map->schedule_type())
                              .first;
    } else {
        new_outer_loop =
            &builder
                 .add_for_after(outer_scope, this->outer_loop_, this->inner_loop_.indvar(),
                                this->inner_loop_.condition(), this->inner_loop_.init(),
                                this->inner_loop_.update())
                 .first;
    }

    // Add new inner loop behind current inner loop
    structured_control_flow::StructuredLoop* new_inner_loop = nullptr;
    if (auto outer_map = dynamic_cast<structured_control_flow::Map*>(&outer_loop_)) {
        new_inner_loop = &builder
                              .add_map_after(inner_scope, this->inner_loop_, outer_map->indvar(),
                                             outer_map->condition(), outer_map->init(),
                                             outer_map->update(), outer_map->schedule_type())
                              .first;
    } else {
        new_inner_loop =
            &builder
                 .add_for_after(inner_scope, this->inner_loop_, this->outer_loop_.indvar(),
                                this->outer_loop_.condition(), this->outer_loop_.init(),
                                this->outer_loop_.update())
                 .first;
    }

    // Insert inner loop body into new inner loop
    auto& inner_body = this->inner_loop_.root();
    builder.insert_children(new_inner_loop->root(), inner_body, 0);

    // Insert outer loop body into new outer loop
    auto& outer_body = this->outer_loop_.root();
    builder.insert_children(new_outer_loop->root(), outer_body, 0);

    // Remove old loops
    builder.remove_child(new_outer_loop->root(), this->inner_loop_);
    builder.remove_child(outer_scope, this->outer_loop_);

    analysis_manager.invalidate_all();
};

void LoopInterchange::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["outer_loop_element_id"] = this->outer_loop_.element_id();
    j["inner_loop_element_id"] = this->inner_loop_.element_id();
};

LoopInterchange LoopInterchange::from_json(builder::StructuredSDFGBuilder& builder,
                                           const nlohmann::json& desc) {
    auto outer_loop_id = desc["outer_loop_element_id"].get<size_t>();
    auto inner_loop_id = desc["inner_loop_element_id"].get<size_t>();
    auto outer_element = builder.find_element_by_id(outer_loop_id);
    auto inner_element = builder.find_element_by_id(inner_loop_id);
    if (!outer_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(outer_loop_id) + " not found.");
    }
    if (!inner_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(inner_loop_id) + " not found.");
    }
    auto outer_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(outer_element);
    auto inner_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(inner_element);

    return LoopInterchange(*outer_loop, *inner_loop);
};

}  // namespace transformations
}  // namespace sdfg
