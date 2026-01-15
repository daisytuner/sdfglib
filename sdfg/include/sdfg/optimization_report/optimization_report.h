#pragma once

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <tuple>

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_sdfg.h"


namespace sdfg {

struct TransformReport {
    bool possible;
    bool applied;
    std::string reason;
    nlohmann::json info;
};

class OptimizationReport {
private:
    StructuredSDFG& sdfg_;
    nlohmann::json report_;
    bool aggregate_;

public:
    OptimizationReport(StructuredSDFG& sdfg, bool aggregate = true);

    void add_pass_entry(const std::string& pass_name, long duration, bool applied);

    void add_transformation_entry(
        int loopnest_index, const std::string& transformation_name, long apply_duration, const TransformReport& report
    );

    void add_target_test(size_t loopnest_index, const std::string& target_name, bool success);

    nlohmann::json get_report();
};

} // namespace sdfg
