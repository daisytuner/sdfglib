#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/arguments_analysis.h>
#include <sstream>

namespace py = pybind11;

/**
 * @brief Python wrapper for ArgumentsAnalysis
 */
class PyArgumentsAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::ArgumentsAnalysis& analysis_;

public:
    PyArgumentsAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::ArgumentsAnalysis>()) {}

    sdfg::analysis::ArgumentsAnalysis& analysis() { return analysis_; }
};
