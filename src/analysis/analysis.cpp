#include "sdfg/analysis/analysis.h"

namespace sdfg {
namespace analysis {

Analysis::Analysis(StructuredSDFG& sdfg)
    : sdfg_(sdfg) {

      };

AnalysisManager::AnalysisManager(StructuredSDFG& sdfg)
    : sdfg_(sdfg) {

      };

AnalysisManager::AnalysisManager(StructuredSDFG& sdfg,
                                 const symbolic::Assumptions& additional_assumptions)
    : sdfg_(sdfg), additional_assumptions_(additional_assumptions) {

      };

void AnalysisManager::invalidate_all() { cache_.clear(); };

}  // namespace analysis
}  // namespace sdfg
