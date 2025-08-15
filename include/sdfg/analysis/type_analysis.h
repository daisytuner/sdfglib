#pragma once

#include "sdfg/analysis/analysis.h"

namespace sdfg {
namespace analysis {

class TypeAnalysis : public Analysis {
private:
    std::unordered_map<std::string, const sdfg::types::IType*> type_map_;

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    TypeAnalysis(StructuredSDFG& sdfg);

    const sdfg::types::IType* get_outer_type(const std::string& container) const;
};

} // namespace analysis
} // namespace sdfg
