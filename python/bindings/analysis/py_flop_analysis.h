#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/flop_analysis.h>

namespace py = pybind11;

/**
 * @brief Python wrapper for FlopAnalysis
 */
class PyFlopAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::FlopAnalysis& analysis_;

public:
    PyFlopAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::FlopAnalysis>()) {}

    sdfg::analysis::FlopAnalysis& analysis() { return analysis_; }
};
