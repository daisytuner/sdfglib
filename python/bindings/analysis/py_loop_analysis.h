#pragma once

#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sstream>
#include "../py_structured_sdfg.h"

namespace py = pybind11;

class PyLoopAnalysis {
private:
    std::unique_ptr<sdfg::analysis::LoopAnalysis> analysis_;

public:
    PyLoopAnalysis(PyStructuredSDFG& sdfg) {
        analysis_ = std::make_unique<sdfg::analysis::LoopAnalysis>(sdfg.sdfg());

        sdfg::analysis::AnalysisManager analysis_manager(sdfg.sdfg());
        analysis_->run(analysis_manager);
    }

    std::optional<sdfg::analysis::LoopInfo> find_loop_by_element_id(size_t element_id) {
        auto loops = this->analysis_->loops();
        for (auto& loop : loops) {
            if (loop->element_id() == element_id) {
                return this->analysis_->loop_info(loop);
            }
        }
        return std::nullopt;
    }

    std::vector<sdfg::analysis::LoopInfo> outermost_loops() {
        auto outermost_loops = this->analysis_->outermost_loops();
        std::vector<sdfg::analysis::LoopInfo> loop_infos;
        for (auto& loop : outermost_loops) {
            loop_infos.push_back(this->analysis_->loop_info(loop));
        }
        return loop_infos;
    }
};

inline void register_loop_analysis(py::module& m) {
    // Register LoopInfo struct
    auto loop_info = py::class_<sdfg::analysis::LoopInfo>(m, "LoopInfo");
#define X(type, name, val) loop_info.def_readwrite(#name, &sdfg::analysis::LoopInfo::name);
    LOOP_INFO_PROPERTIES
#undef X
    loop_info.def("__repr__", [](const sdfg::analysis::LoopInfo& info) {
        std::stringstream ss;
        ss << "<LoopInfo";
#define X(type, name, val) ss << " " << #name << "=" << info.name;
        LOOP_INFO_PROPERTIES
#undef X
        ss << ">";
        return ss.str();
    });

    py::class_<PyLoopAnalysis>(m, "LoopAnalysis")
        .def(py::init<PyStructuredSDFG&>(), py::keep_alive<1, 2>())
        .def("outermost_loops", &PyLoopAnalysis::outermost_loops, "Get LoopInfos for all outermost loops")
        .def(
            "find_loop_by_element_id",
            &PyLoopAnalysis::find_loop_by_element_id,
            py::arg("element_id"),
            "Find LoopInfo by element ID"
        );
}
