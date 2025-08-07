#include "sdfg/optimization_report/opt_report.h"

namespace sdfg {

void OptimizationReport::add_pass_entry_internal(const std::string& pass_name, long duration, bool applied) {}

void OptimizationReport::add_transformation_entry_internal(
    const std::string& transformation_name, long apply_duration, const nlohmann::json& transformation_desc
) {}

OptimizationReport::OptimizationReport(std::vector<std::string> report_types) {}

nlohmann::json OptimizationReport::get_detailed_report_internal() { return detailed_report_; }

nlohmann::json OptimizationReport::get_aggregate_report_internal() { return aggregate_report_; }

void OptimizationReport::add_pass_entry(const std::string& pass_name, long duration, bool applied) {
    instance_->add_pass_entry_internal(pass_name, duration, applied);
}

void OptimizationReport::add_transformation_entry(
    const std::string& transformation_name, long apply_duration, const nlohmann::json& transformation_desc
) {
    instance_->add_transformation_entry_internal(transformation_name, apply_duration, transformation_desc);
}

bool OptimizationReport::applicable() { return instance_ != nullptr; }

void OptimizationReport::initialize(std::vector<std::string> report_types) {
    if (instance_ == nullptr) {
        instance_ = new OptimizationReport(report_types);
    }
}

nlohmann::json OptimizationReport::get_detailed_report() {
    if (instance_ != nullptr) {
        return instance_->get_detailed_report_internal();
    }
    return nlohmann::json();
}

nlohmann::json OptimizationReport::get_aggregate_report() {
    if (instance_ != nullptr) {
        return instance_->get_aggregate_report_internal();
    }
    return nlohmann::json();
}

} // namespace sdfg
