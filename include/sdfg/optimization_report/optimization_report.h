#pragma once

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <tuple>

#include "sdfg/structured_sdfg.h"


namespace sdfg {


class OptimizationReport {
private:
    bool aggregate_;

    std::map<std::string, std::tuple<nlohmann::json, nlohmann::json>> structure_;

    nlohmann::json& get_report_json_internal(const std::string& sdfg_name);

    nlohmann::json& get_loop_list_internal(const std::string& sdfg_name);

    void add_pass_entry_internal(const std::string& pass_name, long duration, bool applied, const std::string& sdfg_name);

    void add_transformation_entry_internal(
        const std::string& transformation_name,
        long apply_duration,
        const nlohmann::json& transformation_desc,
        const std::string& sdfg_name
    );

    OptimizationReport();

    nlohmann::json get_report_internal(const std::string& sdfg_name);

public:
    static void add_pass_entry(const std::string& pass_name, long duration, bool applied, const std::string& sdfg_name);

    static void add_transformation_entry(
        const std::string& transformation_name,
        long apply_duration,
        const nlohmann::json& transformation_desc,
        const std::string& sdfg_name
    );

    static bool applicable();

    static void initialize();

    static nlohmann::json get_report(const std::string& sdfg_name);

    static void add_sdfg_structure(StructuredSDFG& sdfg);

    static void
    add_target_test(const std::string& target_name, const std::string& sdfg_name, size_t loopnest_index, bool success);
};

} // namespace sdfg
