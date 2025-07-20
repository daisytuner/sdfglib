#include "sdfg/transformations/loop_slicing.h"
#include <stdexcept>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace transformations {

enum class LoopSlicingType { Init, Bound, Split_Lt, Split_Le };

LoopSlicing::LoopSlicing(structured_control_flow::StructuredLoop& loop)
    : loop_(loop) {

      };

std::string LoopSlicing::name() const { return "LoopSlicing"; };

bool LoopSlicing::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    if (!analysis::LoopAnalysis::is_contiguous(&loop_, assumptions_analysis)) {
        return false;
    }

    // Collect moving symbols
    std::unordered_set<std::string> moving_symbols;
    auto& all_users = analysis_manager.get<analysis::Users>();
    auto& body = loop_.root();
    analysis::UsersView users(all_users, body);
    for (auto& entry : users.writes()) {
        auto& type = sdfg.type(entry->container());
        if (!dynamic_cast<const types::Scalar*>(&type)) {
            continue;
        }
        if (!types::is_integer(type.primitive_type())) {
            continue;
        }
        moving_symbols.insert(entry->container());
    }

    // Check if loop is sliced by if-elses
    auto indvar = loop_.indvar();
    for (size_t i = 0; i < body.size(); i++) {
        auto child = body.at(i);
        if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&child.first)) {
            if (child.second.assignments().size() > 0) {
                return false;
            }
            if (if_else->size() != 2) {
                return false;
            }

            // Validate condition
            auto branch_1 = if_else->at(0);
            auto condition_1 = branch_1.second;
            if (!symbolic::uses(condition_1, indvar)) {
                return false;
            }
            auto condition_2 = if_else->at(1).second;
            if (!symbolic::eq(condition_1, condition_2->logical_not())) {
                return false;
            }
            for (auto& atom : symbolic::atoms(condition_1)) {
                auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                if (moving_symbols.find(sym->get_name()) != moving_symbols.end()) {
                    return false;
                }
            }
            auto bound = analysis::LoopAnalysis::canonical_bound(&loop_, assumptions_analysis);
            if (bound == SymEngine::null) {
                return false;
            }

            // Case: indvar == init
            if (symbolic::eq(condition_1, symbolic::Eq(indvar, loop_.init()))) {
                return true;
            }

            // Case: indvar == bound - 1
            if (symbolic::eq(condition_1, symbolic::Eq(indvar, symbolic::sub(bound, symbolic::one())))) {
                return true;
            }

            // Case: indvar < new_bound
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*condition_1)) {
                return true;
            }

            // Case: indvar <= new_bound
            if (SymEngine::is_a<SymEngine::LessThan>(*condition_1)) {
                return true;
            }

            return false;
        }
    }

    return false;
};

void LoopSlicing::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();

    auto& body = loop_.root();
    auto indvar = loop_.indvar();

    // Collect loop locals
    auto& users = analysis_manager.get<analysis::Users>();
    auto locals = users.locals(body);

    // Find the if-else that slices the loop
    structured_control_flow::IfElse* if_else = nullptr;
    size_t if_else_index = 0;
    for (size_t i = 0; i < body.size(); i++) {
        auto child = body.at(i);
        auto if_else_ = dynamic_cast<structured_control_flow::IfElse*>(&child.first);
        if (if_else_) {
            if_else_index = i;
            if_else = if_else_;
            break;
        }
    }
    if (if_else == nullptr) {
        throw InvalidSDFGException("LoopSlicing: Expected IfElse");
    }

    auto branch_1 = if_else->at(0);
    auto condition_1 = branch_1.second;
    auto bound = analysis::LoopAnalysis::canonical_bound(&loop_, analysis_manager.get<analysis::AssumptionsAnalysis>());

    LoopSlicingType slice_type = LoopSlicingType::Init;
    if (symbolic::eq(condition_1, symbolic::Eq(indvar, loop_.init()))) {
        slice_type = LoopSlicingType::Init;
    } else if (symbolic::eq(condition_1, symbolic::Eq(indvar, symbolic::sub(bound, symbolic::one())))) {
        slice_type = LoopSlicingType::Bound;
    } else if (SymEngine::is_a<SymEngine::StrictLessThan>(*condition_1)) {
        slice_type = LoopSlicingType::Split_Lt;
    } else if (SymEngine::is_a<SymEngine::LessThan>(*condition_1)) {
        slice_type = LoopSlicingType::Split_Le;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent = static_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(&loop_));

    // Slice loop
    auto indvar_slice_str = builder.find_new_name(indvar->get_name());
    builder.add_container(indvar_slice_str, sdfg.type(indvar->get_name()));
    auto indvar_slice = SymEngine::symbol(indvar_slice_str);
    structured_control_flow::For* loop_slice = nullptr;
    structured_control_flow::For* loop_slice_2 = nullptr;
    switch (slice_type) {
        case LoopSlicingType::Init: {
            auto init_slice = loop_.init();
            auto condition_slice = symbolic::Lt(indvar_slice, symbolic::add(loop_.init(), symbolic::one()));
            auto increment_slice = symbolic::add(indvar_slice, symbolic::one());
            loop_slice =
                &builder.add_for_before(*parent, loop_, indvar_slice, condition_slice, init_slice, increment_slice)
                     .first;
            loop_slice_2 = &builder
                                .add_for_after(
                                    *parent,
                                    loop_,
                                    loop_.indvar(),
                                    loop_.condition(),
                                    symbolic::add(loop_.init(), symbolic::one()),
                                    loop_.update()
                                )
                                .first;
            break;
        }
        case LoopSlicingType::Bound: {
            auto init_slice = symbolic::sub(bound, symbolic::one());
            auto condition_slice = symbolic::subs(loop_.condition(), loop_.indvar(), indvar_slice);
            auto increment_slice = symbolic::add(indvar_slice, symbolic::one());
            loop_slice =
                &builder.add_for_after(*parent, loop_, indvar_slice, condition_slice, init_slice, increment_slice).first;
            loop_slice_2 = &builder
                                .add_for_before(
                                    *parent,
                                    loop_,
                                    loop_.indvar(),
                                    symbolic::Lt(loop_.indvar(), symbolic::sub(loop_.condition(), symbolic::one())),
                                    loop_.init(),
                                    loop_.update()
                                )
                                .first;
            break;
        }
        case LoopSlicingType::Split_Lt: {
            auto init_slice = loop_.init();
            auto condition_slice = symbolic::
                And(symbolic::subs(condition_1, indvar, indvar_slice),
                    symbolic::subs(loop_.condition(), indvar, indvar_slice));
            auto increment_slice = symbolic::add(indvar_slice, symbolic::one());
            loop_slice =
                &builder.add_for_before(*parent, loop_, indvar_slice, condition_slice, init_slice, increment_slice)
                     .first;

            auto condition_bound = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(condition_1);
            auto condition_bound_args = condition_bound->get_args();
            auto condition_bound_args_bound = condition_bound_args.at(0);
            if (symbolic::eq(condition_bound_args_bound, loop_.indvar())) {
                condition_bound_args_bound = condition_bound_args.at(1);
            }

            loop_slice_2 =
                &builder
                     .add_for_after(
                         *parent, loop_, loop_.indvar(), loop_.condition(), condition_bound_args_bound, loop_.update()
                     )
                     .first;
            break;
        }
        case LoopSlicingType::Split_Le: {
            auto init_slice = loop_.init();
            auto condition_slice = symbolic::
                And(symbolic::subs(condition_1, indvar, indvar_slice),
                    symbolic::subs(loop_.condition(), indvar, indvar_slice));
            auto increment_slice = symbolic::add(indvar_slice, symbolic::one());
            loop_slice =
                &builder.add_for_before(*parent, loop_, indvar_slice, condition_slice, init_slice, increment_slice)
                     .first;

            auto condition_bound = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(condition_1);
            auto condition_bound_args = condition_bound->get_args();
            auto condition_bound_args_bound = condition_bound_args.at(0);
            if (symbolic::eq(condition_bound_args_bound, loop_.indvar())) {
                condition_bound_args_bound = condition_bound_args.at(1);
            }

            loop_slice_2 = &builder
                                .add_for_after(
                                    *parent,
                                    loop_,
                                    loop_.indvar(),
                                    loop_.condition(),
                                    symbolic::add(condition_bound_args_bound, symbolic::one()),
                                    loop_.update()
                                )
                                .first;
            break;
        }
    }

    // Move loop locals to the new loop slice
    auto& body_slice = loop_slice->root();

    deepcopy::StructuredSDFGDeepCopy deep_copy(builder, body_slice, body);
    deep_copy.copy();
    auto& body_body_slice = dynamic_cast<structured_control_flow::Sequence&>(body_slice.at(0).first);

    auto& if_else_slice = dynamic_cast<structured_control_flow::IfElse&>(body_body_slice.at(if_else_index).first);
    auto& slice = builder.add_sequence_before(body_body_slice, if_else_slice).first;

    deepcopy::StructuredSDFGDeepCopy deep_copy_slice(builder, slice, if_else_slice.at(0).first);
    deep_copy_slice.copy();

    builder.remove_child(body_body_slice, if_else_index + 1);

    body_body_slice.replace(indvar, indvar_slice);

    // Update remaining loop
    builder.remove_case(*if_else, 0);

    auto& else_slice = builder.add_sequence_before(body, *if_else).first;
    deepcopy::StructuredSDFGDeepCopy deep_copy_else(builder, else_slice, if_else->at(0).first);
    deep_copy_else.copy();

    builder.remove_child(body, if_else_index + 1);

    // Rename all loop-local variables to break artificial dependencies
    for (auto& local : locals) {
        auto new_local = builder.find_new_name(local);
        builder.add_container(new_local, sdfg.type(local));
        loop_slice->root().replace(symbolic::symbol(local), symbolic::symbol(new_local));
    }

    // Move loop locals to the new loop
    builder.insert_children(loop_slice_2->root(), loop_.root(), 0);
    builder.remove_child(*parent, loop_);

    analysis_manager.invalidate_all();
};

void LoopSlicing::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["loop_element_id"] = loop_.element_id();
};

LoopSlicing LoopSlicing::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["loop_element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);

    if (loop == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(loop_id) + " is not a For loop.");
    }

    return LoopSlicing(*loop);
};

} // namespace transformations
} // namespace sdfg
