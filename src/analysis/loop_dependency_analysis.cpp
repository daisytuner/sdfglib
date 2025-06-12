#include "sdfg/analysis/loop_dependency_analysis.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/symbolic/sets.h"

namespace sdfg {
namespace analysis {

LoopDependencyAnalysis::LoopDependencyAnalysis(StructuredSDFG& sdfg)
    : Analysis(sdfg) {

      };

void LoopDependencyAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    this->results_.clear();

    auto& loops_analsis = analysis_manager.get<analysis::LoopAnalysis>();
    for (auto& loop : loops_analsis.loops()) {
        if (auto sloop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
            this->analyze(analysis_manager, sloop);
        }
    }
};

LoopDependencyAnalysisResult LoopDependencyAnalysis::get(
    const structured_control_flow::StructuredLoop& loop) const {
    if (this->results_.find(&loop) == this->results_.end()) {
        return {};
    }
    return this->results_.at(&loop);
};

void LoopDependencyAnalysis::analyze(analysis::AnalysisManager& analysis_manager,
                                     structured_control_flow::StructuredLoop* loop) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Criterion: Strictly monotonic update
    // I.e., the indvar never taskes the same value twice
    if (!loop_analysis.is_monotonic(loop)) {
        return;
    }

    // Criterion: No pointer assignments
    auto& body = loop->root();
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, body);
    if (!body_users.views().empty() || !body_users.moves().empty()) {
        return;
    }

    /*** Dependency Analysis ***/

    this->results_[loop] = LoopDependencyAnalysisResult();
    auto& result = this->results_[loop];

    // Step 1: For loop-carried dependency, the container must be written
    std::unordered_set<std::string> written_containers;
    for (auto& write : body_users.writes()) {
        written_containers.insert(write->container());
    }

    // Step 2: Find all reads potentially causing loop-carried dependency
    std::unordered_map<std::string, std::unordered_set<User*>> open_reads;
    for (auto& container : written_containers) {
        auto reads = body_users.reads(container);

        // Step 3: Filter out reads that are not first uses
        open_reads[container] = std::unordered_set<User*>();
        for (auto& read : reads) {
            // If dominated by a write at the same subset, it is not a first use
            if (body_users.is_dominated_by(*read, Use::WRITE)) {
                continue;
            }
            open_reads[container].insert(read);
        }
    }

    // Step 4: Determine type of dependency
    for (auto& container : written_containers) {
        auto writes = body_users.writes(container);

        // a. Check if reads intersect with writes of other iterations
        if (open_reads.find(container) != open_reads.end()) {
            auto& reads = open_reads[container];
            for (auto& read : reads) {
                for (auto& write : writes) {
                    if (this->intersects(read, write)) {
                        result[container] = LoopCarriedDependency::RAW;
                        break;
                    }
                }
                if (result[container] == LoopCarriedDependency::RAW) {
                    break;
                }
            }
        }

        // For a RAW dependency, we are done
        if (result[container] == LoopCarriedDependency::RAW) {
            continue;
        }

        // b. Check if writes intersect with writes of other iterations
        for (auto& write : writes) {
            for (auto& write_other : writes) {
                if (this->intersects(write, write_other)) {
                    result[container] = LoopCarriedDependency::WAR;
                    break;
                }
            }
            if (result[container] == LoopCarriedDependency::WAR) {
                break;
            }
        }
    }

    return;
}

bool LoopDependencyAnalysis::intersects(User* first, User* second) const {
    /*
    for (auto& subset : first->subsets()) {
        for (auto& subset_other : second->subsets()) {
            if (symbolic::intersect(subset, subset_other)) {
                return true;
            }
        }
    }
    return false;
    */
    return true;
}

}  // namespace analysis
}  // namespace sdfg
