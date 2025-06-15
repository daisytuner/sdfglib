#include "sdfg/analysis/loop_dependency_analysis.h"

#include "sdfg/analysis/assumptions_analysis.h"
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

    LoopDependencyAnalysisResult result;

    // Step 0: Get assumptions
    auto& assumptions = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto body_assums = assumptions.get(body);

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
        for (auto& read : reads) {
            // If dominated by a write at the same subset, it is not a first use
            if (body_users.is_dominated_by(*read, Use::WRITE, body_assums)) {
                continue;
            }
            if (open_reads.find(read->container()) == open_reads.end()) {
                open_reads[read->container()] = {};
            }
            open_reads[read->container()].insert(read);
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
                    if (this->intersects(read, write, *loop, body_users, analysis_manager)) {
                        result[container] = LoopCarriedDependency::RAW;
                        break;
                    }
                }
                if (result.find(container) != result.end() &&
                    result.at(container) == LoopCarriedDependency::RAW) {
                    break;
                }
            }
        }

        // For a RAW dependency, we are done
        if (result.find(container) != result.end() &&
            result.at(container) == LoopCarriedDependency::RAW) {
            continue;
        }

        // b. Check if writes intersect with writes of other iterations
        for (auto& write : writes) {
            for (auto& write_other : writes) {
                if (this->intersects(write, write_other, *loop, body_users, analysis_manager)) {
                    result[container] = LoopCarriedDependency::WAW;
                    break;
                }
            }
            if (result.find(container) != result.end() &&
                result.at(container) == LoopCarriedDependency::WAW) {
                break;
            }
        }
    }

    this->results_[loop] = result;
}

bool LoopDependencyAnalysis::intersects(User* first, User* second,
                                        structured_control_flow::StructuredLoop& loop,
                                        analysis::UsersView& body_users,
                                        analysis::AnalysisManager& analysis_manager) const {
    // Try to find tighter assumptions by lowering the scope
    auto scope_first = analysis::Users::scope(first);
    auto scope_second = analysis::Users::scope(second);
    // Simplification: If the scopes are different, we conservatively assume the loop body as the
    // scope
    if (scope_first != scope_second) {
        scope_first = &loop.root();
    }
    auto& assumptions = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto scope_assums = assumptions.get(*scope_first);

    for (auto& subset : first->subsets()) {
        // Collect all symbols of the subset;
        symbolic::SymbolSet first_symbols;
        for (auto& dim : subset) {
            for (auto& symbol : symbolic::atoms(dim)) {
                first_symbols.insert(symbol);
            }
        }

        for (auto& subset_other : second->subsets()) {
            // Collect all symbols of the subset;
            symbolic::SymbolSet second_symbols;
            for (auto& dim : subset_other) {
                for (auto& symbol : symbolic::atoms(dim)) {
                    second_symbols.insert(symbol);
                }
            }

            // Collect all symbols of the subsets;
            symbolic::SymbolSet symbols;
            for (auto& symbol : first_symbols) {
                symbols.insert(symbol);
            }
            for (auto& symbol : second_symbols) {
                symbols.insert(symbol);
            }

            // Determine which symbols are readonly -> params
            symbolic::SymbolSet params;
            for (auto& symbol : symbols) {
                if (symbol == loop.indvar()) {
                    continue;
                }
                if (body_users.writes(symbol->get_name()).empty()) {
                    params.insert(symbol);
                }
            }

            if (!symbolic::is_disjoint(subset, subset_other, params, {loop.indvar()},
                                       scope_assums)) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace analysis
}  // namespace sdfg
