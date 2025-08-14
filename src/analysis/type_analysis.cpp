#include "sdfg/analysis/type_analysis.h"
#include "sdfg/analysis/users.h"

namespace sdfg {
namespace analysis {

void TypeAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    for (auto container : this->sdfg_.containers()) {
        if (sdfg_.type(container).type_id() == sdfg::types::TypeID::Pointer) {
            auto pointer_type = static_cast<const sdfg::types::Pointer*>(&sdfg_.type(container));
            if (!pointer_type->has_pointee_type()) {
                continue;
            }
        }
        type_map_.insert({container, &sdfg_.type(container)});
    }

    auto& users = analysis_manager.get<Users>();

    // TODO: iterate over writes

    // TODO: iterate over reads

    // TODO: iterate over views

    // TODO: iterate over moves
}

TypeAnalysis::TypeAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

const std::unordered_map<std::string, const sdfg::types::IType*>& TypeAnalysis::type_map() const { return type_map_; }

} // namespace analysis
} // namespace sdfg
