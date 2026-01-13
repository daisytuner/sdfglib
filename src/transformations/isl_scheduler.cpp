#include "sdfg/transformations/isl_scheduler.h"

#include <isl/constraint.h>
#include <isl/map.h>
#include <isl/schedule_node.h>
#include <isl/space.h>
#include <isl/union_map.h>

namespace sdfg {
namespace transformations {

ISLScheduler::ISLScheduler(structured_control_flow::StructuredLoop& loop)
    : loop_(loop), scop_(nullptr), dependences_(nullptr) {};

std::string ISLScheduler::name() const { return "ISLScheduler"; };

bool ISLScheduler::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    analysis::ScopBuilder scop_builder(builder.subject(), loop_);
    this->scop_ = scop_builder.build(analysis_manager);
    if (!this->scop_) {
        return false;
    }
    this->dependences_ = std::make_unique<analysis::Dependences>(*this->scop_);
    if (!this->dependences_->has_valid_dependences()) {
        this->scop_ = nullptr;
        this->dependences_ = nullptr;
        return false;
    }

    return true;
};

void ISLScheduler::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    assert(
        this->scop_ != nullptr && this->dependences_ != nullptr &&
        "ISLScheduler::apply called without successful can_be_applied"
    );

    int validity_kinds = analysis::Dependences::TYPE_RAW | analysis::Dependences::TYPE_WAR |
                         analysis::Dependences::TYPE_WAW;
    int proximity_kinds = analysis::Dependences::TYPE_RAW | analysis::Dependences::TYPE_WAR |
                          analysis::Dependences::TYPE_WAW;
    isl_union_map* validity = this->dependences_->dependences(validity_kinds);
    isl_union_map* proximity = this->dependences_->dependences(proximity_kinds);

    isl_union_set* domain = this->scop_->domains();
    validity = isl_union_map_gist_domain(validity, isl_union_set_copy(domain));
    validity = isl_union_map_gist_range(validity, isl_union_set_copy(domain));
    proximity = isl_union_map_gist_domain(proximity, isl_union_set_copy(domain));
    proximity = isl_union_map_gist_range(proximity, isl_union_set_copy(domain));

    // Add spatial proximity
    isl_union_map* spatial_proximity = isl_union_map_empty(isl_union_map_get_space(proximity));

    for (auto* stmt : this->scop_->statements()) {
        for (auto* access : stmt->accesses()) {
            isl_map* rel = access->relation();
            if (isl_map_dim(rel, isl_dim_out) == 0) {
                continue;
            }

            // We create a map that maps each element to the next element in the last dimension
            isl_space* array_space = isl_space_range(isl_map_get_space(rel));
            isl_map* adj = isl_map_universe(isl_space_map_from_set(isl_space_copy(array_space)));

            int n_dim = isl_map_dim(adj, isl_dim_out);
            for (int i = 0; i < n_dim - 1; ++i) {
                adj = isl_map_equate(adj, isl_dim_in, i, isl_dim_out, i);
            }

            isl_local_space* ls = isl_local_space_from_space(isl_map_get_space(adj));
            isl_constraint* c = isl_equality_alloc(ls);
            c = isl_constraint_set_coefficient_si(c, isl_dim_out, n_dim - 1, 1);
            c = isl_constraint_set_coefficient_si(c, isl_dim_in, n_dim - 1, -1);
            c = isl_constraint_set_constant_si(c, -1);
            adj = isl_map_add_constraint(adj, c);

            // S -> Array
            isl_map* rel_map = isl_map_copy(rel);
            // S_next -> Array_next = Array + 1
            // We want S -> S_next such that Acc(S_next) = Acc(S) + 1
            // S -> Acc(S) -> Acc(S)+1 -> S_next
            // Proximity = Rel . Adj . Rel^-1

            isl_map* map = isl_map_apply_range(rel_map, adj);
            map = isl_map_apply_range(map, isl_map_reverse(isl_map_copy(rel)));

            spatial_proximity = isl_union_map_add_map(spatial_proximity, map);

            isl_space_free(array_space);
        }
    }

    proximity = isl_union_map_union(proximity, spatial_proximity);

    int isl_maximize_bands = 1;
    int isl_outer_coincidence = 0;

    isl_options_set_schedule_outer_coincidence(scop_->ctx(), isl_outer_coincidence);
    isl_options_set_schedule_maximize_band_depth(scop_->ctx(), isl_maximize_bands);
    isl_options_set_schedule_max_constant_term(scop_->ctx(), 20);
    isl_options_set_schedule_max_coefficient(scop_->ctx(), 20);
    isl_options_set_tile_scale_tile_loops(scop_->ctx(), 0);

    isl_schedule_constraints* SC = isl_schedule_constraints_on_domain(isl_union_set_copy(domain));
    SC = isl_schedule_constraints_set_proximity(SC, isl_union_map_copy(proximity));
    SC = isl_schedule_constraints_set_validity(SC, isl_union_map_copy(validity));
    SC = isl_schedule_constraints_set_coincidence(SC, isl_union_map_copy(validity));
    isl_schedule* S = isl_schedule_constraints_compute_schedule(SC);
    scop_->set_schedule_tree(S);
    isl_schedule_free(S);

    auto& sdfg = builder.subject();
    analysis::ScopToSDFG converter(*scop_, builder);
    converter.build(analysis_manager);

    this->applied_ = true;
    this->scop_ = nullptr;
    this->dependences_ = nullptr;
};

void ISLScheduler::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw InvalidSDFGException("Unsupported loop type for serialization of loop: " + loop_.indvar()->get_name());
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["parameters"] = {};
};

ISLScheduler ISLScheduler::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);

    return ISLScheduler(*loop);
};

} // namespace transformations
} // namespace sdfg
