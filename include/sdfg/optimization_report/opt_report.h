#pragma once

#include <nlohmann/json.hpp>


namespace sdfg {


class OptimizationReport {
private:
    bool detailed;
    bool aggregate;

    nlohmann::json aggregate_report_;
    nlohmann::json detailed_report_;

    static OptimizationReport* instance_;

    void add_pass_entry_internal(const std::string& pass_name, long duration, bool applied);

    void add_transformation_entry_internal(
        const std::string& transformation_name, long apply_duration, const nlohmann::json& transformation_desc
    );

    OptimizationReport(std::vector<std::string> report_types);

    nlohmann::json get_detailed_report_internal();

    nlohmann::json get_aggregate_report_internal();

public:
    static void add_pass_entry(const std::string& pass_name, long duration, bool applied);

    static void add_transformation_entry(
        const std::string& transformation_name, long apply_duration, const nlohmann::json& transformation_desc
    );

    static bool applicable();

    static void initialize(std::vector<std::string> report_types);

    static nlohmann::json get_detailed_report();

    static nlohmann::json get_aggregate_report();
};

} // namespace sdfg
