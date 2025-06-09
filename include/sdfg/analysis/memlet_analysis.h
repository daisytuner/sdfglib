#pragma once

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>

#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

/**
 * Data Parallelism Classes
 * - DEPENDENT: No parallelism (loop-carried dependencies)
 * - REDUCTION: Point-wise parallelism (reductions)
 * - PARALLEL: Full parallelism w.r.t. the induction variable
 * - PRIVATE: A new version of the container is created in each iteration
 * - READONLY: Read-only containers
 */
enum Parallelism { DEPENDENT, REDUCTION, PARALLEL, PRIVATE, READONLY };

typedef std::unordered_map<std::string, Parallelism> MemletAnalysisResult;

class MemletAnalysis : public Analysis {
   private:
    std::unordered_set<const structured_control_flow::StructuredLoop*> loops_;
    std::unordered_map<const structured_control_flow::StructuredLoop*,
                       MemletAnalysisResult>
        results_;

    bool disjoint(const data_flow::Subset& subset1, const data_flow::Subset& subset2,
                  const std::string& indvar, const std::unordered_set<std::string>& moving_symbols,
                  const symbolic::Assumptions& assumptions);

    void classify(analysis::AnalysisManager& analysis_manager,
                  const structured_control_flow::StructuredLoop* loop);

   protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

   public:
    MemletAnalysis(StructuredSDFG& sdfg);

    const MemletAnalysisResult& get(
        const structured_control_flow::StructuredLoop& loop) const;

    static std::pair<data_flow::Subset, data_flow::Subset> substitution(
        const data_flow::Subset& subset1, const data_flow::Subset& subset2,
        const std::string& indvar, const std::unordered_set<std::string>& moving_symbols,
        symbolic::SymbolMap& replacements, std::vector<std::string>& substitions);

    static std::pair<data_flow::Subset, data_flow::Subset> delinearization(
        const data_flow::Subset& subset1, const data_flow::Subset& subset2,
        const std::unordered_set<std::string>& moving_symbols,
        const symbolic::Assumptions& assumptions);
};

}  // namespace analysis
}  // namespace sdfg