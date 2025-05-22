#include "sdfg/passes/structured_control_flow/for2map.h"

#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/basic.h"
#include "symengine/logic.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace passes {

For2Map::For2Map()
    : Pass() {

      };

std::string For2Map::name() { return "For2Map"; };

symbolic::Expression For2Map::num_iterations(const structured_control_flow::For& for_stmt,
                                             analysis::AnalysisManager& analysis_manager) const {
    // Criterion: loop must be data-parallel w.r.t containers
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& dependencies = analysis.get(for_stmt);
    if (dependencies.size() == 0) {
        return symbolic::zero();
    }

    // Criterion: update must be normalizable (i.e., it may not be involved in anything but an
    // addition during the update)
    auto& index_var = for_stmt.indvar();
    auto& update = for_stmt.update();

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
        auto condition_LE = SymEngine::rcp_dynamic_cast<SymEngine::LessThan>(condition);
        lhs = condition_LE->get_arg1();
        rhs = condition_LE->get_arg2();
    } else if (SymEngine::is_a<SymEngine::StrictLessThan>(*condition)) {
        auto condition_LT = SymEngine::rcp_dynamic_cast<SymEngine::StrictLessThan>(condition);
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

bool For2Map::run_pass(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();

    auto& root = sdfg.root();
    if (root.size() == 0) {
        return false;
    }

    std::list<structured_control_flow::Sequence*> queue = {&sdfg.root()};
    while (!queue.empty()) {
        auto curr = queue.front();
        queue.pop_front();

        for (size_t i = 0; i < curr->size(); i++) {
            auto& child = curr->at(i).first;

            if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(&child)) {
                // Add to queue
                for (size_t j = 0; j < if_else_stmt->size(); j++) {
                    queue.push_back(&if_else_stmt->at(j).first);
                }
            } else if (auto loop_stmt = dynamic_cast<structured_control_flow::While*>(&child)) {
                auto& root = loop_stmt->root();
                queue.push_back(&root);
            } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(&child)) {
                auto num_iterations = this->num_iterations(*for_stmt, analysis_manager);
                if (symbolic::eq(num_iterations, symbolic::zero())) {
                    auto& root = for_stmt->root();
                    queue.push_back(&root);
                    continue;
                }

                // Create map
                auto& map = builder.convert_for(*curr, *for_stmt, num_iterations);
                auto& root = map.root();
                auto stride =
                    symbolic::subs(for_stmt->update(), for_stmt->indvar(), symbolic::zero());

                auto replacement =
                    symbolic::add(symbolic::mul(map.indvar(), stride), for_stmt->init());
                root.replace(for_stmt->indvar(), replacement);

                queue.push_back(&root);
                applied = true;

            } else if (auto kernel_stmt = dynamic_cast<structured_control_flow::Kernel*>(&child)) {
                auto& root = kernel_stmt->root();
                queue.push_back(&root);
            } else if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(&child)) {
                auto& root = map_stmt->root();
                queue.push_back(&root);
            }
        }
    }

    return applied;
}

}  // namespace passes
}  // namespace sdfg
