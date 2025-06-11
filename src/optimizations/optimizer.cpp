#include <sdfg/optimizations/optimizer.h>
#include <sdfg/transformations/loop_distribute.h>
#include <sdfg/transformations/loop_interchange.h>
#include <sdfg/transformations/loop_slicing.h>
#include <sdfg/transformations/loop_tiling.h>
#include <sdfg/transformations/out_local_storage.h>

namespace sdfg {
namespace optimizations {

Optimizer::Optimizer() : history_(nlohmann::json::array()) {}

void Optimizer::replay(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager,
                       const nlohmann::json& transformation_data) {
    std::cout << "Replaying transformations..." << std::endl;
    std::cout << "Transformation data: " << transformation_data.dump(4) << std::endl;

    for (const auto& desc : transformation_data) {
        std::cout << "Processing transformation description: " << std::endl;
        auto transformation_name = desc["transformation_type"];
        std::cout << "Applying transformation: " << transformation_name.get<std::string>()
                  << std::endl;

        if (transformation_name == "LoopTiling") {
            this->apply<transformations::LoopTiling>(builder, analysis_manager, desc);
        } else if (transformation_name == "LoopDistribute") {
            this->apply<transformations::LoopDistribute>(builder, analysis_manager, desc);
        } else if (transformation_name == "LoopInterchange") {
            this->apply<transformations::LoopInterchange>(builder, analysis_manager, desc);
        } else if (transformation_name == "LoopSlicing") {
            this->apply<transformations::LoopSlicing>(builder, analysis_manager, desc);
        } else if (transformation_name == "OutLocalStorage") {
            this->apply<transformations::OutLocalStorage>(builder, analysis_manager, desc);
        } else {
            throw transformations::InvalidTransformationDescriptionException(
                "Unknown transformation: " + transformation_name.get<std::string>());
        }
    }
}

void Optimizer::save(std::filesystem::path path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving transformations: " +
                                 path.string());
    }
    std::cout << "Saving transformations to " << path.string() << std::endl;
    std::cout << history_.dump(4) << std::endl;  // Pretty print with an indent of 4 spaces
    file << history_.dump(4);                    // Pretty print with an indent of 4 spaces
    file.close();
}

}  // namespace optimizations
}  // namespace sdfg