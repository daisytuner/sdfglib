#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include <sdfg/analysis/users.h>

namespace py = pybind11;

/**
 * @brief Python wrapper for Users analysis
 */
class PyUsers {
private:
    sdfg::analysis::AnalysisManager& manager_;
    sdfg::analysis::Users& analysis_;

public:
    PyUsers(sdfg::analysis::AnalysisManager& manager)
        : manager_(manager), analysis_(manager.get<sdfg::analysis::Users>()) {}

    sdfg::analysis::Users& analysis() { return analysis_; }
};
