#include "sdfg/transformations/parallelization.h"
#include <stdexcept>

namespace sdfg {
namespace transformations {

Parallelization::Parallelization(structured_control_flow::Map& map) : map_(map) {}

std::string Parallelization::name() const { return "Parallelization"; }

bool Parallelization::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return map_.schedule_type().value() == structured_control_flow::ScheduleType_Sequential::value();
}

void Parallelization::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    this->map_.schedule_type() = structured_control_flow::ScheduleType_CPU_Parallel();
}

void Parallelization::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->map_.element_id()}, {"type", "map"}}}};
    j["transformation_type"] = this->name();
}

Parallelization Parallelization::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto map_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(map_id);
    if (element == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(map_id) + " not found.");
    }

    auto loop = dynamic_cast<structured_control_flow::Map*>(element);

    if (loop == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(map_id) + " is not a Map.");
    }

    return Parallelization(*loop);
}

} // namespace transformations
} // namespace sdfg
