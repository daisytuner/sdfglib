#include "sdfg/optimization_report/optimization_report.h"

#include <nlohmann/json_fwd.hpp>
#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/map.h"


namespace sdfg {

static OptimizationReport* instance_;

void OptimizationReport::add_pass_entry_internal(const std::string& pass_name, long duration, bool applied) {}

void OptimizationReport::add_transformation_entry_internal(
    const std::string& transformation_name, long apply_duration, const nlohmann::json& transformation_desc
) {}

OptimizationReport::OptimizationReport() {}

nlohmann::json OptimizationReport::get_detailed_report_internal() {
    nlohmann::json detailed_report;
    detailed_report["Timings"] = detailed_report_;
    detailed_report["Loops"] = outermost_loops_;
    detailed_report["Maps"] = outermost_maps_;
    return detailed_report;
}

nlohmann::json OptimizationReport::get_aggregate_report_internal() {
    nlohmann::json aggregate_report;
    aggregate_report["Timings"] = aggregate_report_;
    aggregate_report["Loops"] = outermost_loops_;
    aggregate_report["Maps"] = outermost_maps_;
    return aggregate_report;
}

void OptimizationReport::add_pass_entry(const std::string& pass_name, long duration, bool applied) {
    instance_->add_pass_entry_internal(pass_name, duration, applied);
}

void OptimizationReport::add_transformation_entry(
    const std::string& transformation_name, long apply_duration, const nlohmann::json& transformation_desc
) {
    instance_->add_transformation_entry_internal(transformation_name, apply_duration, transformation_desc);
}

bool OptimizationReport::applicable() { return instance_ != nullptr; }

void OptimizationReport::initialize() {
    if (instance_ == nullptr) {
        instance_ = new OptimizationReport();
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

void OptimizationReport::add_sdfg_structure(StructuredSDFG& sdfg) {
    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    // Add outermost loops
    instance_->outermost_loops_ = nlohmann::json::array();

    for (int i = 0; i < loop_analysis.outermost_loops().size(); i++) {
        nlohmann::json loop_json;
        auto loop = loop_analysis.outermost_loops().at(i);
        loop_json["loopnest_index"] = i;
        loop_json["element_id"] = loop->element_id();
        serializer::JSONSerializer serializer;
        nlohmann::json debug_info;
        serializer.debug_info_to_json(debug_info, loop->debug_info());
        loop_json["debug_info"] = debug_info;

        if (auto while_loop = dynamic_cast<const structured_control_flow::While*>(loop)) {
            loop_json["type"] = "while";
        } else if (auto for_loop = dynamic_cast<const structured_control_flow::For*>(loop)) {
            loop_json["type"] = "for";
        } else if (auto map = dynamic_cast<const structured_control_flow::Map*>(loop)) {
            loop_json["type"] = "map";
        }
        instance_->outermost_loops_.push_back(loop_json);
    }

    // Add outermost maps
    instance_->outermost_maps_ = nlohmann::json::array();
    for (int i = 0; i < loop_analysis.outermost_maps().size(); i++) {
        nlohmann::json map_json;
        auto map = loop_analysis.outermost_maps().at(i);
        map_json["mapnest_index"] = i;
        map_json["element_id"] = map->element_id();
        serializer::JSONSerializer serializer;
        nlohmann::json debug_info;
        serializer.debug_info_to_json(debug_info, map->debug_info());
        map_json["debug_info"] = debug_info;
        instance_->outermost_maps_.push_back(map_json);
    }
}

void OptimizationReport::add_target_test(std::string target_name, size_t mapnest_index, bool success) {
    if (!instance_->outermost_maps_.is_array()) {
        instance_->outermost_maps_ = nlohmann::json::array();
    }
    
    instance_->outermost_maps_.at(mapnest_index)["target_name"] = target_name;
    instance_->outermost_maps_.at(mapnest_index)["success"] = success;
}

void OptimizationReport::add_global_timestamp(std::string name, long timestamp) {
    instance_->aggregate_report_[name] = timestamp;
    instance_->detailed_report_[name] = timestamp;
}

} // namespace sdfg
