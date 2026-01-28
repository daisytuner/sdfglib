#include "py_tasklet.h"

#include <sdfg/data_flow/tasklet.h>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace sdfg::types;

void register_tasklet(py::module& m) {
    // TaskletCode enum
    py::enum_<sdfg::data_flow::TaskletCode>(m, "TaskletCode")
        .value("assign", sdfg::data_flow::TaskletCode::assign)
        // Floating-point operations
        .value("fp_neg", sdfg::data_flow::TaskletCode::fp_neg)
        .value("fp_add", sdfg::data_flow::TaskletCode::fp_add)
        .value("fp_sub", sdfg::data_flow::TaskletCode::fp_sub)
        .value("fp_mul", sdfg::data_flow::TaskletCode::fp_mul)
        .value("fp_div", sdfg::data_flow::TaskletCode::fp_div)
        .value("fp_rem", sdfg::data_flow::TaskletCode::fp_rem)
        .value("fp_fma", sdfg::data_flow::TaskletCode::fp_fma)
        // Floating-point comparisons
        .value("fp_oeq", sdfg::data_flow::TaskletCode::fp_oeq)
        .value("fp_one", sdfg::data_flow::TaskletCode::fp_one)
        .value("fp_oge", sdfg::data_flow::TaskletCode::fp_oge)
        .value("fp_ogt", sdfg::data_flow::TaskletCode::fp_ogt)
        .value("fp_ole", sdfg::data_flow::TaskletCode::fp_ole)
        .value("fp_olt", sdfg::data_flow::TaskletCode::fp_olt)
        .value("fp_ord", sdfg::data_flow::TaskletCode::fp_ord)
        .value("fp_ueq", sdfg::data_flow::TaskletCode::fp_ueq)
        .value("fp_une", sdfg::data_flow::TaskletCode::fp_une)
        .value("fp_ugt", sdfg::data_flow::TaskletCode::fp_ugt)
        .value("fp_uge", sdfg::data_flow::TaskletCode::fp_uge)
        .value("fp_ult", sdfg::data_flow::TaskletCode::fp_ult)
        .value("fp_ule", sdfg::data_flow::TaskletCode::fp_ule)
        .value("fp_uno", sdfg::data_flow::TaskletCode::fp_uno)
        // Integer operations
        .value("int_add", sdfg::data_flow::TaskletCode::int_add)
        .value("int_sub", sdfg::data_flow::TaskletCode::int_sub)
        .value("int_mul", sdfg::data_flow::TaskletCode::int_mul)
        .value("int_sdiv", sdfg::data_flow::TaskletCode::int_sdiv)
        .value("int_srem", sdfg::data_flow::TaskletCode::int_srem)
        .value("int_udiv", sdfg::data_flow::TaskletCode::int_udiv)
        .value("int_urem", sdfg::data_flow::TaskletCode::int_urem)
        .value("int_and", sdfg::data_flow::TaskletCode::int_and)
        .value("int_or", sdfg::data_flow::TaskletCode::int_or)
        .value("int_xor", sdfg::data_flow::TaskletCode::int_xor)
        .value("int_shl", sdfg::data_flow::TaskletCode::int_shl)
        .value("int_ashr", sdfg::data_flow::TaskletCode::int_ashr)
        .value("int_lshr", sdfg::data_flow::TaskletCode::int_lshr)
        .value("int_smin", sdfg::data_flow::TaskletCode::int_smin)
        .value("int_smax", sdfg::data_flow::TaskletCode::int_smax)
        .value("int_scmp", sdfg::data_flow::TaskletCode::int_scmp)
        .value("int_umin", sdfg::data_flow::TaskletCode::int_umin)
        .value("int_umax", sdfg::data_flow::TaskletCode::int_umax)
        .value("int_ucmp", sdfg::data_flow::TaskletCode::int_ucmp)
        // Integer comparisons
        .value("int_eq", sdfg::data_flow::TaskletCode::int_eq)
        .value("int_ne", sdfg::data_flow::TaskletCode::int_ne)
        .value("int_sge", sdfg::data_flow::TaskletCode::int_sge)
        .value("int_sgt", sdfg::data_flow::TaskletCode::int_sgt)
        .value("int_sle", sdfg::data_flow::TaskletCode::int_sle)
        .value("int_slt", sdfg::data_flow::TaskletCode::int_slt)
        .value("int_uge", sdfg::data_flow::TaskletCode::int_uge)
        .value("int_ugt", sdfg::data_flow::TaskletCode::int_ugt)
        .value("int_ule", sdfg::data_flow::TaskletCode::int_ule)
        .value("int_ult", sdfg::data_flow::TaskletCode::int_ult)
        .value("int_abs", sdfg::data_flow::TaskletCode::int_abs)
        .export_values();
}
