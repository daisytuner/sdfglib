#include "sdfg/analysis/analysis.h"

namespace sdfg {
namespace analysis {

Analysis::Analysis(StructuredSDFG& sdfg)
    : sdfg_(sdfg) {

      };

AnalysisManager::AnalysisManager(StructuredSDFG& sdfg)
    : sdfg_(sdfg) {

      };

void AnalysisManager::invalidate_all() { cache_.clear(); };

} // namespace analysis
} // namespace sdfg
