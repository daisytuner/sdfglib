#include "sdfg/transformations/parallelization.h"

namespace sdfg {
namespace transformations {

Parallelization::Parallelization(structured_control_flow::Map& map) : map_(map) {}

std::string Parallelization::name() const { return "Parallelization"; }

bool Parallelization::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return map_.schedule_type() == structured_control_flow::ScheduleType_Sequential;
}

void Parallelization::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    this->map_.schedule_type() = structured_control_flow::ScheduleType_CPU_Parallel;
}

void Parallelization::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["map_element_id"] = map_.element_id();
}

Parallelization Parallelization::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto map_id = desc["map_element_id"].get<size_t>();
    auto element = builder.find_element_by_id(map_id);
    return Parallelization(*dynamic_cast<structured_control_flow::Map*>(element));
}

} // namespace transformations
} // namespace sdfg
