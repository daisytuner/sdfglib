#include "sdfg/passes/structured_control_flow/for2map.h"

#include "sdfg/analysis/memlet_analysis.h"

namespace sdfg {
namespace passes {

For2Map::For2Map(builder::StructuredSDFGBuilder& builder,
                 analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {

      };

bool For2Map::can_be_applied(const structured_control_flow::For& for_stmt,
                             analysis::AnalysisManager& analysis_manager) {
    // Criterion: loop must be data-parallel w.r.t containers
    auto& analysis = analysis_manager.get<analysis::MemletAnalysis>();
    auto& dependencies = analysis.get(for_stmt);
    if (dependencies.size() == 0) {
        return false;
    }

    // Criterion: update must be normalizable (i.e., it may not be involved in anything but an
    // addition during the update)
    auto& index_var = for_stmt.indvar();
    auto& update = for_stmt.update();

    bool normalizable_update = symbolic::eq(
        symbolic::subs(update, index_var, symbolic::one()),
        symbolic::add(symbolic::subs(update, index_var, symbolic::zero()), symbolic::one()));

    if (!normalizable_update) {
        return false;
    }

    auto stride = symbolic::subs(update, index_var, symbolic::zero());

    // Criterion: loop bound must be simple, less than or equal statement (e.g., i < N)

    auto condition = symbolic::rearrange_simple_condition(for_stmt.condition(), index_var);

    symbolic::Expression bound;
    symbolic::Expression lhs;
    symbolic::Expression rhs;
    if (SymEngine::is_a<SymEngine::LessThan>(*condition)) {
        auto condition_LE = SymEngine::rcp_dynamic_cast<const SymEngine::LessThan>(condition);
        lhs = condition_LE->get_arg1();
        rhs = condition_LE->get_arg2();
    } else if (SymEngine::is_a<SymEngine::StrictLessThan>(*condition)) {
        auto condition_LT = SymEngine::rcp_dynamic_cast<const SymEngine::StrictLessThan>(condition);
        lhs = condition_LT->get_arg1();
        rhs = condition_LT->get_arg2();
    } else {
        return false;
    }

    if (!symbolic::eq(lhs, index_var)) {
        return false;
    }

    return true;
}

symbolic::Expression For2Map::num_iterations(const structured_control_flow::For& for_stmt,
                                             analysis::AnalysisManager& analysis_manager) const {
    // Criterion: update must be normalizable (i.e., it may not be involved in anything but an
    // addition during the update)
    auto& index_var = for_stmt.indvar();
    auto& update = for_stmt.update();
    auto& init = for_stmt.init();

    bool normalizable_update = symbolic::eq(
        symbolic::subs(update, index_var, symbolic::one()),
        symbolic::add(symbolic::subs(update, index_var, symbolic::zero()), symbolic::one()));

    if (!normalizable_update) {
        return symbolic::zero();
    }

    auto stride = symbolic::subs(update, index_var, symbolic::zero());

    // Criterion: loop bound must be simple, less than or equal statement (e.g., i < N)

    auto condition = symbolic::rearrange_simple_condition(for_stmt.condition(), index_var);

    symbolic::Expression bound;
    bool is_strict = false;
    symbolic::Expression lhs;
    symbolic::Expression rhs;
    if (SymEngine::is_a<SymEngine::LessThan>(*condition)) {
        auto condition_LE = SymEngine::rcp_dynamic_cast<const SymEngine::LessThan>(condition);
        lhs = condition_LE->get_arg1();
        rhs = condition_LE->get_arg2();
    } else if (SymEngine::is_a<SymEngine::StrictLessThan>(*condition)) {
        auto condition_LT = SymEngine::rcp_dynamic_cast<const SymEngine::StrictLessThan>(condition);
        lhs = condition_LT->get_arg1();
        rhs = condition_LT->get_arg2();
        is_strict = true;
    } else {
        return symbolic::zero();
    }

    if (symbolic::eq(lhs, index_var)) {
        bound = rhs;
    } else {
        return symbolic::zero();
    }

    if (!is_strict) {
        bound = symbolic::add(bound, symbolic::one());
    }

    // subtract the init value from the bound
    bound = symbolic::sub(bound, init);

    symbolic::Expression num_iterations;

    if (symbolic::eq(stride, symbolic::one())) {
        num_iterations = bound;
    } else if (symbolic::eq(stride, symbolic::zero())) {
        throw std::runtime_error("Stride is zero");
    } else {
        num_iterations = symbolic::ceil(symbolic::div(bound, stride));
    }

    return num_iterations;
}

void For2Map::apply(structured_control_flow::For& for_stmt, builder::StructuredSDFGBuilder& builder,
                    analysis::AnalysisManager& analysis_manager) {
    auto num_iterations = this->num_iterations(for_stmt, analysis_manager);

    auto init = for_stmt.init();
    auto indvar = for_stmt.indvar();
    auto update = for_stmt.update();

    auto& parent = builder.parent(for_stmt);

    // Create map
    auto& map = builder.convert_for(parent, for_stmt, num_iterations);
    auto& root = map.root();
    auto stride = symbolic::subs(update, indvar, symbolic::zero());

    auto replacement = symbolic::add(symbolic::mul(map.indvar(), stride), init);
    root.replace(map.indvar(), replacement);

    auto successor = builder.add_block_after(parent, map);
    successor.second.assignments().insert({indvar, num_iterations});
}

bool For2Map::accept(structured_control_flow::Sequence& parent,
                     structured_control_flow::For& node) {
    if (!this->can_be_applied(node, analysis_manager_)) {
        return false;
    }

    this->apply(node, builder_, analysis_manager_);
    return true;
}

}  // namespace passes
}  // namespace sdfg
