#include <sdfg/transformations/loop_distribute.h>
#include <sdfg/transformations/loop_interchange.h>
#include <sdfg/transformations/loop_slicing.h>
#include <sdfg/transformations/loop_tiling.h>
#include <sdfg/transformations/out_local_storage.h>
#include <sdfg/transformations/recorder.h>

namespace sdfg {
namespace transformations {

Recorder::Recorder() : history_(nlohmann::json::array()) {}

void Recorder::replay(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    const nlohmann::json& transformation_data,
    bool skip_if_not_applicable
) {
    for (const auto& desc : transformation_data) {
        auto transformation_name = desc["transformation_type"];

        if (transformation_name == "LoopTiling") {
            this->apply<transformations::LoopTiling>(builder, analysis_manager, desc, skip_if_not_applicable);
        } else if (transformation_name == "LoopDistribute") {
            this->apply<transformations::LoopDistribute>(builder, analysis_manager, desc, skip_if_not_applicable);
        } else if (transformation_name == "LoopInterchange") {
            this->apply<transformations::LoopInterchange>(builder, analysis_manager, desc, skip_if_not_applicable);
        } else if (transformation_name == "LoopSlicing") {
            this->apply<transformations::LoopSlicing>(builder, analysis_manager, desc, skip_if_not_applicable);
        } else if (transformation_name == "OutLocalStorage") {
            this->apply<transformations::OutLocalStorage>(builder, analysis_manager, desc, skip_if_not_applicable);
        } else {
            throw transformations::InvalidTransformationDescriptionException(
                "Unknown transformation: " + transformation_name.get<std::string>()
            );
        }
    }
}

void Recorder::save(std::filesystem::path path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving transformations: " + path.string());
    }
    file << history_.dump(4); // Pretty print with an indent of 4 spaces
    file.close();
}

} // namespace transformations
} // namespace sdfg
