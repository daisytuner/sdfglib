#include "sdfg/passes/schedules/dok_scheduling_pass.h"
#include <thread>
#include "sdfg/analysis/degrees_of_knowledge_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/symbolic/symbolic.h"


namespace sdfg {
namespace passes {

DOKScheduling::DOKScheduling() {}

std::string DOKScheduling::name() { return "DOKScheduling"; }

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
        if (load.second == analysis::DegreesOfKnowledgeClassification::Unbound) {
            size_threshold = symbolic::one();
        } else {
            auto load_first = symbolic::max(load.first, symbolic::one());
            size_threshold = symbolic::max(symbolic::one(), symbolic::div(load_threshold, load_first));
        }

        symbolic::Expression num_threads;
        if (size.second == analysis::DegreesOfKnowledgeClassification::Unbound) {
            num_threads = symbolic::integer(avail_threads);
        } else {
            num_threads = symbolic::min(symbolic::div(size.first, size_threshold), symbolic::integer(avail_threads));
        }

        symbolic::Condition run_dynamic;
        if (balance.second == analysis::DegreesOfKnowledgeClassification::Scalar) {
            run_dynamic = symbolic::__false__();
        } else if (balance.second == analysis::DegreesOfKnowledgeClassification::Bound) {
            /* if (balance_threshold > symbolic::atoms(balance.first).size()) {
                run_dynamic = symbolic::__false__();
            } */
            run_dynamic = symbolic::__true__();
        } else {
            run_dynamic = symbolic::__true__();
        }

        ScheduleType schedule_type = ScheduleType_CPU_Parallel::create();
        if (!symbolic::is_false(run_dynamic)) {
            ScheduleType_CPU_Parallel::omp_schedule(schedule_type, OpenMPSchedule::Dynamic);
            ScheduleType_CPU_Parallel::num_threads(schedule_type, symbolic::integer(avail_threads));
        } else {
            ScheduleType_CPU_Parallel::num_threads(schedule_type, symbolic::max(symbolic::one(), num_threads));
        }

        if (symbolic::is_false(
                symbolic::Le(symbolic::div(symbolic::mul(size.first, load.first), number.first), number_threshold)
            )) {
            ScheduleType_CPU_Parallel::num_threads(schedule_type, symbolic::integer(1));
        }

        builder.update_schedule_type(*map, schedule_type);
    }

    return false;
}

void DOKScheduling::read_thresholds() {
    // Read thresholds from configuration or set default values
    load_threshold = symbolic::integer(100);
    balance_threshold = 50;
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
            balance_threshold = std::stoi(balance_env);
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
