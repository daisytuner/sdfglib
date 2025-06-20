#include "sdfg/transformations/loop_distribute.h"

#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"

namespace sdfg {
namespace transformations {

LoopDistribute::LoopDistribute(structured_control_flow::StructuredLoop& loop)
    : loop_(loop) {

      };

std::string LoopDistribute::name() const { return "LoopDistribute"; };

bool LoopDistribute::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                    analysis::AnalysisManager& analysis_manager) {
    auto indvar = this->loop_.indvar();

    // Criterion: Block -> Loop
    auto& body = this->loop_.root();
    if (body.size() < 2) {
        return false;
    }
    auto& block = body.at(0).first;
    if (!body.at(0).second.assignments().empty()) {
        return false;
    }

    auto& users = analysis_manager.get<analysis::Users>();

    // Determine block-related containers
    std::unordered_set<std::string> containers;
    analysis::UsersView block_users(users, block);
    for (auto& user : block_users.uses()) {
        containers.insert(user->container());
    }

    // Criterion: loop is data-parallel w.r.t containers
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop_);
    if (dependencies.size() == 0) {
        return false;
    }

    // Determine body- and block-local variables
    auto body_locals = users.locals(body);
    auto block_locals = users.locals(block);

    analysis::UsersView body_users(users, body);

    // Check if all dependencies can be resolved
    bool can_be_distributed = true;
    for (auto& dep : dependencies) {
        auto& container = dep.first;
        auto& dep_type = dep.second;

        // Criterion: If container not used in block, ignore
        if (containers.find(container) == containers.end()) {
            continue;
        }

        // Criterion: Containers must be parallel
        if (dep_type < analysis::Parallelism::PARALLEL) {
            can_be_distributed = false;
            break;
        }

        // Criterion: Readonly and parallel containers -> no action
        if (dep_type == analysis::Parallelism::READONLY ||
            dep_type == analysis::Parallelism::PARALLEL) {
            continue;
        }

        // We are left with private containers

        // Criterion: If container is only used inside block, no action
        if (block_locals.find(container) != block_locals.end()) {
            continue;
        }
        // Criterion: If container is used outside the loop, we fail
        if (body_locals.find(container) == body_locals.end()) {
            can_be_distributed = false;
            break;
        }

        // Criterion: Container must only be used as access node
        for (auto& user : body_users.uses(container)) {
            if (dynamic_cast<data_flow::AccessNode*>(user->element()) == nullptr) {
                can_be_distributed = false;
                break;
            }
        }
        if (!can_be_distributed) {
            break;
        }

        // Criterion: Bound must be integer
        auto bound = analysis::DataParallelismAnalysis::bound(this->loop_);
        if (bound == SymEngine::null || !SymEngine::is_a<SymEngine::Integer>(*bound)) {
            can_be_distributed = false;
            break;
        }
    }
    if (!can_be_distributed) {
        return false;
    }

    return true;
};

void LoopDistribute::apply(builder::StructuredSDFGBuilder& builder,
                           analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    auto indvar = this->loop_.indvar();
    auto condition = this->loop_.condition();
    auto update = this->loop_.update();
    auto init = this->loop_.init();

    auto& body = this->loop_.root();
    auto& block = body.at(0).first;

    // We might need to extend containers to loop dimension

    auto& users = analysis_manager.get<analysis::Users>();
    auto body_locals = users.locals(body);
    auto block_locals = users.locals(block);

    analysis::UsersView body_users(users, body);
    analysis::UsersView block_users(users, block);

    // Determine block-related containers
    std::unordered_set<std::string> containers;
    for (auto& user : block_users.uses()) {
        containers.insert(user->container());
    }

    std::unordered_set<std::string> shared_containers;
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(loop_);
    for (auto& dep : dependencies) {
        auto& container = dep.first;
        auto& dep_type = dep.second;

        if (containers.find(container) == containers.end()) {
            continue;
        }
        if (dep_type == analysis::Parallelism::READONLY ||
            dep_type == analysis::Parallelism::PARALLEL) {
            continue;
        }
        if (block_locals.find(container) != block_locals.end()) {
            continue;
        }

        shared_containers.insert(container);
    }

    if (!shared_containers.empty()) {
        auto bound = analysis::DataParallelismAnalysis::bound(this->loop_);
        for (auto& shared_container : shared_containers) {
            auto& type = sdfg.type(shared_container);

            // Add loop dimension to subset
            for (auto& user : body_users.uses(shared_container)) {
                auto& access_node = static_cast<data_flow::AccessNode&>(*user->element());
                auto& graph = access_node.get_parent();
                for (auto& edge : graph.in_edges(access_node)) {
                    data_flow::Subset new_subset = {indvar};
                    if (!dynamic_cast<const types::Scalar*>(&type)) {
                        for (auto& dim : edge.subset()) {
                            new_subset.push_back(dim);
                        }
                    }
                    edge.subset() = new_subset;
                }
                for (auto& edge : graph.out_edges(access_node)) {
                    data_flow::Subset new_subset = {indvar};
                    if (!dynamic_cast<const types::Scalar*>(&type)) {
                        for (auto& dim : edge.subset()) {
                            new_subset.push_back(dim);
                        }
                    }
                    edge.subset() = new_subset;
                }
            }

            // Make array
            builder.make_array(shared_container, bound);
        }
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent =
        static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&loop_));

    // Copy loop
    auto& new_loop =
        builder.add_for_before(*parent, this->loop_, indvar, condition, init, update).first;

    auto& new_body = new_loop.root();
    deepcopy::StructuredSDFGDeepCopy copies(builder, new_body, block);
    copies.copy();

    // Replace indvar in new loop
    std::string new_indvar = builder.find_new_name(indvar->get_name());
    builder.add_container(new_indvar, sdfg.type(indvar->get_name()));
    new_loop.replace(indvar, symbolic::symbol(new_indvar));

    // Remove block from loop
    builder.remove_child(body, block);

    analysis_manager.invalidate_all();
};

void LoopDistribute::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["loop_element_id"] = loop_.element_id();
};

LoopDistribute LoopDistribute::from_json(builder::StructuredSDFGBuilder& builder,
                                         const nlohmann::json& desc) {
    auto loop_id = desc["loop_element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " +
                                                        std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::For*>(element);

    return LoopDistribute(*loop);
};

}  // namespace transformations
}  // namespace sdfg
