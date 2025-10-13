#pragma once

#include <nlohmann/json.hpp>

#include "sdfg/structured_sdfg.h"


namespace sdfg {


class OptimizationReport {
private:
    bool detailed;
    bool aggregate;

    nlohmann::json aggregate_report_;
    nlohmann::json detailed_report_;
    nlohmann::json outermost_loops_;
    nlohmann::json outermost_maps_;

    void add_pass_entry_internal(const std::string& pass_name, long duration, bool applied);

    void add_transformation_entry_internal(
        const std::string& transformation_name, long apply_duration, const nlohmann::json& transformation_desc
    );

    OptimizationReport();

    nlohmann::json get_detailed_report_internal();

    nlohmann::json get_aggregate_report_internal();

public:
    static void add_pass_entry(const std::string& pass_name, long duration, bool applied);

    static void add_transformation_entry(
        const std::string& transformation_name, long apply_duration, const nlohmann::json& transformation_desc
    );

    static bool applicable();

    static void initialize();

    static nlohmann::json get_detailed_report();

    static nlohmann::json get_aggregate_report();

    static void add_sdfg_structure(StructuredSDFG& sdfg);

    static void add_target_test(std::string target_name, size_t mapnest_index, bool success);

    static void add_global_timestamp(std::string name, long timestamp);
};

} // namespace sdfg
