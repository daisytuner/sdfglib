#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/escape_analysis.h>

namespace py = pybind11;

/**
 * @brief Python wrapper for EscapeAnalysis
 */
class PyEscapeAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::EscapeAnalysis& analysis_;

public:
    PyEscapeAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::EscapeAnalysis>()) {}

    sdfg::analysis::EscapeAnalysis& analysis() { return analysis_; }
};
