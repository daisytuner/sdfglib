#pragma once

#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sstream>

namespace py = pybind11;

/**
 * @brief Python wrapper for LoopAnalysis
 */
class PyLoopAnalysis {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::LoopAnalysis& analysis_;

public:
    PyLoopAnalysis(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::LoopAnalysis>()) {}

    sdfg::analysis::LoopAnalysis& analysis() { return analysis_; }
};
