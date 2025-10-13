#include "sdfg/optimization_report/optimization_report.h"

#include <nlohmann/json_fwd.hpp>
#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/map.h"


namespace sdfg {

static OptimizationReport* instance_;

void OptimizationReport::add_pass_entry_internal(const std::string& pass_name, long duration, bool applied, const std::string& sdfg_name) {
    // TODO
}

void OptimizationReport::add_transformation_entry_internal(
    const std::string& transformation_name, long apply_duration, const nlohmann::json& transformation_desc, const std::string& sdfg_name
) {
    // TODO
}

OptimizationReport::OptimizationReport() {}

nlohmann::json& OptimizationReport::get_report_json_internal(const std::string& sdfg_name) {
    return std::get<0>(structure_[sdfg_name]);
}

nlohmann::json& OptimizationReport::get_loop_list_internal(const std::string& sdfg_name) {
    return std::get<1>(structure_[sdfg_name]);
}

nlohmann::json& OptimizationReport::get_map_list_internal(const std::string& sdfg_name) {
    return std::get<2>(structure_[sdfg_name]);
}

nlohmann::json OptimizationReport::get_report_internal(const std::string& sdfg_name) {
    nlohmann::json report;
    report["Timings"] = get_report_json_internal(sdfg_name);
    report["Loops"] = get_loop_list_internal(sdfg_name);
    report["Maps"] = get_map_list_internal(sdfg_name);
    return report;
}

void OptimizationReport::add_pass_entry(const std::string& pass_name, long duration, bool applied, const std::string& sdfg_name) {
    instance_->add_pass_entry_internal(pass_name, duration, applied, sdfg_name);
}

void OptimizationReport::add_transformation_entry(
    const std::string& transformation_name, long apply_duration, const nlohmann::json& transformation_desc, const std::string& sdfg_name
) {
    instance_->add_transformation_entry_internal(transformation_name, apply_duration, transformation_desc, sdfg_name);
}

bool OptimizationReport::applicable() { return instance_ != nullptr; }

void OptimizationReport::initialize() {
    if (instance_ == nullptr) {
        instance_ = new OptimizationReport();
    }
}

nlohmann::json OptimizationReport::get_report(const std::string& sdfg_name) {
    if (instance_ != nullptr) {
        return instance_->get_report_internal(sdfg_name);
    }
    return nlohmann::json();
}

void OptimizationReport::add_sdfg_structure(StructuredSDFG& sdfg) {
    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    // Add outermost loops
    instance_->get_loop_list_internal(sdfg.name()) = nlohmann::json::array();

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
        instance_->get_loop_list_internal(sdfg.name()).push_back(loop_json);
    }

    // Add outermost maps
    instance_->get_map_list_internal(sdfg.name()) = nlohmann::json::array();
    for (int i = 0; i < loop_analysis.outermost_maps().size(); i++) {
        nlohmann::json map_json;
        auto map = loop_analysis.outermost_maps().at(i);
        map_json["mapnest_index"] = i;
        map_json["element_id"] = map->element_id();
        serializer::JSONSerializer serializer;
        nlohmann::json debug_info;
        serializer.debug_info_to_json(debug_info, map->debug_info());
        map_json["debug_info"] = debug_info;
        instance_->get_map_list_internal(sdfg.name()).push_back(map_json);
    }
}

void OptimizationReport::add_target_test(const std::string& target_name, const std::string& sdfg_name, size_t mapnest_index, bool success) {
    instance_->get_map_list_internal(sdfg_name).at(mapnest_index)["target_name"] = target_name;
    instance_->get_map_list_internal(sdfg_name).at(mapnest_index)["success"] = success;
}

} // namespace sdfg
