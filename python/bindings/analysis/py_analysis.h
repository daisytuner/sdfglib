#pragma once

#include <memory>
#include <optional>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sdfg/analysis/analysis.h>

#include "py_structured_sdfg.h"

#include "py_arguments_analysis.h"
#include "py_assumptions_analysis.h"
#include "py_control_flow_analysis.h"
#include "py_dominance_analysis.h"
#include "py_escape_analysis.h"
#include "py_flop_analysis.h"
#include "py_loop_analysis.h"
#include "py_scope_analysis.h"
#include "py_type_analysis.h"
#include "py_users.h"

namespace py = pybind11;

/**
 * @brief Python wrapper for the AnalysisManager
 *
 * This class provides a Python-friendly interface to the analysis manager,
 * allowing users to query analyses in a similar style to the C++ API.
 * Analysis objects are cached and reused to ensure identity comparison works.
 */
class PyAnalysisManager {
private:
    PyStructuredSDFG& sdfg_;
    std::unique_ptr<sdfg::analysis::AnalysisManager> manager_;

    // Cached analysis wrappers
    std::optional<PyArgumentsAnalysis> arguments_analysis_;
    std::optional<PyAssumptionsAnalysis> assumptions_analysis_;
    std::optional<PyControlFlowAnalysis> control_flow_analysis_;
    std::optional<PyDominanceAnalysis> dominance_analysis_;
    std::optional<PyEscapeAnalysis> escape_analysis_;
    std::optional<PyFlopAnalysis> flop_analysis_;
    std::optional<PyLoopAnalysis> loop_analysis_;
    std::optional<PyScopeAnalysis> scope_analysis_;
    std::optional<PyTypeAnalysis> type_analysis_;
    std::optional<PyUsers> users_;

public:
    PyAnalysisManager(PyStructuredSDFG& sdfg)
        : sdfg_(sdfg), manager_(std::make_unique<sdfg::analysis::AnalysisManager>(sdfg.sdfg())) {}

    sdfg::analysis::AnalysisManager& manager() { return *manager_; }

    PyStructuredSDFG& sdfg() { return sdfg_; }

    void invalidate_all() {
        // Clear cached wrappers
        arguments_analysis_.reset();
        assumptions_analysis_.reset();
        control_flow_analysis_.reset();
        dominance_analysis_.reset();
        escape_analysis_.reset();
        flop_analysis_.reset();
        loop_analysis_.reset();
        scope_analysis_.reset();
        type_analysis_.reset();
        users_.reset();

        manager_->invalidate_all();
    }

    PyArgumentsAnalysis& arguments_analysis() {
        if (!arguments_analysis_) {
            arguments_analysis_.emplace(*manager_);
        }
        return *arguments_analysis_;
    }

    PyAssumptionsAnalysis& assumptions_analysis() {
        if (!assumptions_analysis_) {
            assumptions_analysis_.emplace(*manager_);
        }
        return *assumptions_analysis_;
    }

    PyControlFlowAnalysis& control_flow_analysis() {
        if (!control_flow_analysis_) {
            control_flow_analysis_.emplace(*manager_);
        }
        return *control_flow_analysis_;
    }

    PyDominanceAnalysis& dominance_analysis() {
        if (!dominance_analysis_) {
            dominance_analysis_.emplace(*manager_);
        }
        return *dominance_analysis_;
    }

    PyEscapeAnalysis& escape_analysis() {
        if (!escape_analysis_) {
            escape_analysis_.emplace(*manager_);
        }
        return *escape_analysis_;
    }

    PyFlopAnalysis& flop_analysis() {
        if (!flop_analysis_) {
            flop_analysis_.emplace(*manager_);
        }
        return *flop_analysis_;
    }

    PyLoopAnalysis& loop_analysis() {
        if (!loop_analysis_) {
            loop_analysis_.emplace(*manager_);
        }
        return *loop_analysis_;
    }

    PyScopeAnalysis& scope_analysis() {
        if (!scope_analysis_) {
            scope_analysis_.emplace(*manager_);
        }
        return *scope_analysis_;
    }

    PyTypeAnalysis& type_analysis() {
        if (!type_analysis_) {
            type_analysis_.emplace(*manager_);
        }
        return *type_analysis_;
    }

    PyUsers& users() {
        if (!users_) {
            users_.emplace(*manager_);
        }
        return *users_;
    }
};

inline void register_analysis(py::module& m) {
    py::class_<PyAnalysisManager>(m, "AnalysisManager")
        .def(
            py::init<PyStructuredSDFG&>(),
            py::keep_alive<1, 2>(),
            py::arg("sdfg"),
            "Create an AnalysisManager for the given SDFG"
        )
        .def("invalidate_all", &PyAnalysisManager::invalidate_all, "Invalidate all cached analyses")
        .def(
            "arguments_analysis",
            &PyAnalysisManager::arguments_analysis,
            py::return_value_policy::reference_internal,
            "Get the ArgumentsAnalysis"
        )
        .def(
            "assumptions_analysis",
            &PyAnalysisManager::assumptions_analysis,
            py::return_value_policy::reference_internal,
            "Get the AssumptionsAnalysis"
        )
        .def(
            "control_flow_analysis",
            &PyAnalysisManager::control_flow_analysis,
            py::return_value_policy::reference_internal,
            "Get the ControlFlowAnalysis"
        )
        .def(
            "dominance_analysis",
            &PyAnalysisManager::dominance_analysis,
            py::return_value_policy::reference_internal,
            "Get the DominanceAnalysis"
        )
        .def(
            "escape_analysis",
            &PyAnalysisManager::escape_analysis,
            py::return_value_policy::reference_internal,
            "Get the EscapeAnalysis"
        )
        .def(
            "flop_analysis",
            &PyAnalysisManager::flop_analysis,
            py::return_value_policy::reference_internal,
            "Get the FlopAnalysis"
        )
        .def(
            "loop_analysis",
            &PyAnalysisManager::loop_analysis,
            py::return_value_policy::reference_internal,
            "Get the LoopAnalysis"
        )
        .def(
            "scope_analysis",
            &PyAnalysisManager::scope_analysis,
            py::return_value_policy::reference_internal,
            "Get the ScopeAnalysis"
        )
        .def(
            "type_analysis",
            &PyAnalysisManager::type_analysis,
            py::return_value_policy::reference_internal,
            "Get the TypeAnalysis"
        )
        .def("users", &PyAnalysisManager::users, py::return_value_policy::reference_internal, "Get the Users analysis")
        .def("__repr__", [](const PyAnalysisManager&) { return "<AnalysisManager>"; });

    py::class_<PyArgumentsAnalysis>(m, "ArgumentsAnalysis").def("__repr__", [](const PyArgumentsAnalysis&) {
        return "<ArgumentsAnalysis>";
    });

    py::class_<PyAssumptionsAnalysis>(m, "AssumptionsAnalysis").def("__repr__", [](const PyAssumptionsAnalysis&) {
        return "<AssumptionsAnalysis>";
    });

    py::class_<PyScopeAnalysis>(m, "ScopeAnalysis").def("__repr__", [](const PyScopeAnalysis&) {
        return "<ScopeAnalysis>";
    });

    py::class_<PyControlFlowAnalysis>(m, "ControlFlowAnalysis").def("__repr__", [](const PyControlFlowAnalysis&) {
        return "<ControlFlowAnalysis>";
    });

    py::class_<PyDominanceAnalysis>(m, "DominanceAnalysis").def("__repr__", [](const PyDominanceAnalysis&) {
        return "<DominanceAnalysis>";
    });

    py::class_<PyEscapeAnalysis>(m, "EscapeAnalysis").def("__repr__", [](const PyEscapeAnalysis&) {
        return "<EscapeAnalysis>";
    });

    py::class_<PyFlopAnalysis>(m, "FlopAnalysis").def("__repr__", [](const PyFlopAnalysis&) {
        return "<FlopAnalysis>";
    });

    py::class_<PyLoopAnalysis>(m, "LoopAnalysis").def("__repr__", [](const PyLoopAnalysis&) {
        return "<LoopAnalysis>";
    });

    py::class_<PyTypeAnalysis>(m, "TypeAnalysis").def("__repr__", [](const PyTypeAnalysis&) {
        return "<TypeAnalysis>";
    });

    py::class_<PyUsers>(m, "Users").def("__repr__", [](const PyUsers&) { return "<Users>"; });
}
