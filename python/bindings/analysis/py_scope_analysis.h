#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/scope_analysis.h>

namespace py = pybind11;

/**
 * @brief Python wrapper for ScopeAnalysis
 */
class PyScopeAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::ScopeAnalysis& analysis_;

public:
    PyScopeAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::ScopeAnalysis>()) {}

    sdfg::analysis::ScopeAnalysis& analysis() { return analysis_; }
};
