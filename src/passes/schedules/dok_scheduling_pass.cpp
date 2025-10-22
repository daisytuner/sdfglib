#include "sdfg/passes/schedules/dok_scheduling_pass.h"
#include <thread>
#include "sdfg/analysis/degrees_of_knowledge_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/dict.h"


namespace sdfg {
namespace passes {

DOKScheduling::DOKScheduling() {}

std::string DOKScheduling::name() { return "DOKScheduling"; }

void dynamic_balance_branch(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    Map& map_node,
    symbolic::Condition run_dynamic
) {
    auto num_threads = ScheduleType_CPU_Parallel::num_threads(map_node.schedule_type());
    if (symbolic::eq(num_threads, symbolic::one())) {
        return;
    }

    auto& scope = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent = scope.parent_scope(&map_node);

    auto parent_sequence = static_cast<Sequence*>(parent);

    auto& new_sequence = builder.add_sequence_after(*parent_sequence, map_node);

    size_t id = -1;
    for (int i = 0; i < parent_sequence->size(); i++) {
        if (map_node.element_id() == parent_sequence->at(i).first.element_id()) {
            for (auto assignment : parent_sequence->at(i).second.assignments()) {
                parent_sequence->at(i + 1).second.assignments()[assignment.first] = assignment.second;
            }
            id = i;
            break;
        }
    }

    auto& branch = builder.add_if_else(new_sequence);

    auto& then_case = builder.add_case(branch, run_dynamic);
    auto& else_case = builder.add_case(branch, symbolic::__true__());

    deepcopy::StructuredSDFGDeepCopy copier_then(builder, then_case, map_node);
    copier_then.copy();

    auto then_map = static_cast<Map*>(&then_case.at(0).first);

    deepcopy::StructuredSDFGDeepCopy copier_else(builder, else_case, map_node);
    auto else_mapping = copier_else.copy();

    auto else_map = static_cast<Map*>(&else_case.at(0).first);

    builder.remove_child(*parent_sequence, id);

    ScheduleType then_schedule_type = ScheduleType_CPU_Parallel::create();
    ScheduleType else_schedule_type = ScheduleType_CPU_Parallel::create();
    ScheduleType_CPU_Parallel::num_threads(then_schedule_type, num_threads);
    ScheduleType_CPU_Parallel::num_threads(else_schedule_type, num_threads);

    ScheduleType_CPU_Parallel::omp_schedule(then_schedule_type, OpenMPSchedule::Dynamic);

    builder.update_schedule_type(*then_map, then_schedule_type);
    builder.update_schedule_type(*else_map, else_schedule_type);
}

bool DOKScheduling::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& dok_analysis = analysis_manager.get<analysis::DegreesOfKnowledgeAnalysis>();

    auto outermost_maps = loop_analysis.outermost_maps();

    read_thresholds();

    // Schedule outermost maps
    for (const auto& node : outermost_maps) {
        Map* map = static_cast<Map*>(node);
        // test for dynamic behaviour
        auto balance = dok_analysis.balance_of_a_map(*map);
        auto load = dok_analysis.load_of_a_map(*map);
        auto size = dok_analysis.size_of_a_map(*map);
        auto number = dok_analysis.number_of_maps(*map);

        // compute size threshold on demand
        symbolic::Expression load_first;
        if (load.second == analysis::DegreesOfKnowledgeClassification::Unbound ||
            load.second == analysis::DegreesOfKnowledgeClassification::Bound) {
            load_first = symbolic::subs(load.first, map->indvar(), symbolic::div(size.first, symbolic::integer(2)));
            for (auto atom : symbolic::atoms(load_first)) {
                load_first = symbolic::subs(load_first, atom, symbolic::one());
            }
        } else {
            load_first = symbolic::max(load.first, symbolic::one());
        }
        size_threshold = symbolic::max(symbolic::one(), symbolic::div(load_threshold, load_first));

        symbolic::Expression num_threads;
        num_threads = symbolic::
            max(symbolic::one(),
                symbolic::min(symbolic::div(size.first, size_threshold), symbolic::integer(avail_threads)));

        ScheduleType schedule_type = ScheduleType_CPU_Parallel::create();
        if (balance.second == analysis::DegreesOfKnowledgeClassification::Unbound) {
            // num_threads = symbolic::integer(avail_threads);
            ScheduleType_CPU_Parallel::omp_schedule(schedule_type, OpenMPSchedule::Dynamic);
        }

        /* if (number.second == analysis::DegreesOfKnowledgeClassification::Unbound ||
            number.second == analysis::DegreesOfKnowledgeClassification::Bound) {
            num_threads = symbolic::one();
        } else if (!symbolic::eq(number.first, symbolic::one())) {
            num_threads = symbolic::one();
        } */

        ScheduleType_CPU_Parallel::num_threads(schedule_type, num_threads);

        builder.update_schedule_type(*map, schedule_type);

        if ((size.second == analysis::DegreesOfKnowledgeClassification::Bound ||
             size.second == analysis::DegreesOfKnowledgeClassification::Unbound) &&
            (balance.second == analysis::DegreesOfKnowledgeClassification::Bound)) {
            auto balance_bound = symbolic::Ge(size.first, symbolic::mul(num_threads, balance_threshold));

            dynamic_balance_branch(builder, analysis_manager, *map, balance_bound);
        }

        if (number.second == analysis::DegreesOfKnowledgeClassification::Bound) {
            // TODO: generate new branch with static and dynamic schedules
        }
    }

    return false;
}

void DOKScheduling::read_thresholds() {
    // Read thresholds from configuration or set default values
    load_threshold = symbolic::integer(100);
    balance_threshold = symbolic::integer(50);
    size_threshold = symbolic::integer(200); // Example value
    number_threshold = symbolic::integer(10); // Example value

    const char* threshold_env = std::getenv("DOK_LOAD_THRESHOLD");
    if (threshold_env) {
        try {
            load_threshold = symbolic::integer(std::stoi(threshold_env));
        } catch (const std::exception&) {
        }
    }

    const char* balance_env = std::getenv("DOK_BALANCE_THRESHOLD");
    if (balance_env) {
        try {
            balance_threshold = symbolic::integer(std::stoi(balance_env));
        } catch (const std::exception&) {
        }
    }

    const char* number_env = std::getenv("DOK_NUMBER_THRESHOLD");
    if (number_env) {
        try {
            number_threshold = symbolic::integer(std::stoi(number_env));
        } catch (const std::exception&) {
        }
    }

    const char* threads = std::getenv("DOK_NUM_THREADS");
    if (threads) {
        try {
            avail_threads = std::stoi(threads);
        } catch (const std::exception&) {
        }
    } else {
        avail_threads = std::thread::hardware_concurrency();
    }
}

} // namespace passes
} // namespace sdfg
