#include "py_cmath.h"

#include <sdfg/data_flow/library_nodes/math/cmath/cmath_node.h>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_cmath(py::module& m) {
    // CMathFunction enum
    py::enum_<sdfg::math::cmath::CMathFunction>(m, "CMathFunction")
        // Trigonometric functions
        .value("sin", sdfg::math::cmath::CMathFunction::sin)
        .value("cos", sdfg::math::cmath::CMathFunction::cos)
        .value("tan", sdfg::math::cmath::CMathFunction::tan)
        .value("asin", sdfg::math::cmath::CMathFunction::asin)
        .value("acos", sdfg::math::cmath::CMathFunction::acos)
        .value("atan", sdfg::math::cmath::CMathFunction::atan)
        .value("atan2", sdfg::math::cmath::CMathFunction::atan2)
        // Hyperbolic functions
        .value("sinh", sdfg::math::cmath::CMathFunction::sinh)
        .value("cosh", sdfg::math::cmath::CMathFunction::cosh)
        .value("tanh", sdfg::math::cmath::CMathFunction::tanh)
        .value("asinh", sdfg::math::cmath::CMathFunction::asinh)
        .value("acosh", sdfg::math::cmath::CMathFunction::acosh)
        .value("atanh", sdfg::math::cmath::CMathFunction::atanh)
        // Exponential and logarithmic functions
        .value("exp", sdfg::math::cmath::CMathFunction::exp)
        .value("exp2", sdfg::math::cmath::CMathFunction::exp2)
        .value("exp10", sdfg::math::cmath::CMathFunction::exp10)
        .value("expm1", sdfg::math::cmath::CMathFunction::expm1)
        .value("log", sdfg::math::cmath::CMathFunction::log)
        .value("log10", sdfg::math::cmath::CMathFunction::log10)
        .value("log2", sdfg::math::cmath::CMathFunction::log2)
        .value("log1p", sdfg::math::cmath::CMathFunction::log1p)
        // Power functions
        .value("pow", sdfg::math::cmath::CMathFunction::pow)
        .value("sqrt", sdfg::math::cmath::CMathFunction::sqrt)
        .value("cbrt", sdfg::math::cmath::CMathFunction::cbrt)
        .value("hypot", sdfg::math::cmath::CMathFunction::hypot)
        // Error and gamma functions
        .value("erf", sdfg::math::cmath::CMathFunction::erf)
        .value("erfc", sdfg::math::cmath::CMathFunction::erfc)
        .value("tgamma", sdfg::math::cmath::CMathFunction::tgamma)
        .value("lgamma", sdfg::math::cmath::CMathFunction::lgamma)
        // Rounding and remainder functions
        .value("fabs", sdfg::math::cmath::CMathFunction::fabs)
        .value("ceil", sdfg::math::cmath::CMathFunction::ceil)
        .value("floor", sdfg::math::cmath::CMathFunction::floor)
        .value("trunc", sdfg::math::cmath::CMathFunction::trunc)
        .value("round", sdfg::math::cmath::CMathFunction::round)
        .value("lround", sdfg::math::cmath::CMathFunction::lround)
        .value("llround", sdfg::math::cmath::CMathFunction::llround)
        .value("roundeven", sdfg::math::cmath::CMathFunction::roundeven)
        .value("nearbyint", sdfg::math::cmath::CMathFunction::nearbyint)
        .value("rint", sdfg::math::cmath::CMathFunction::rint)
        .value("lrint", sdfg::math::cmath::CMathFunction::lrint)
        .value("llrint", sdfg::math::cmath::CMathFunction::llrint)
        .value("fmod", sdfg::math::cmath::CMathFunction::fmod)
        .value("remainder", sdfg::math::cmath::CMathFunction::remainder)
        // Floating-point manipulation functions
        .value("frexp", sdfg::math::cmath::CMathFunction::frexp)
        .value("ldexp", sdfg::math::cmath::CMathFunction::ldexp)
        .value("modf", sdfg::math::cmath::CMathFunction::modf)
        .value("scalbn", sdfg::math::cmath::CMathFunction::scalbn)
        .value("scalbln", sdfg::math::cmath::CMathFunction::scalbln)
        .value("ilogb", sdfg::math::cmath::CMathFunction::ilogb)
        .value("logb", sdfg::math::cmath::CMathFunction::logb)
        .value("nextafter", sdfg::math::cmath::CMathFunction::nextafter)
        .value("nexttoward", sdfg::math::cmath::CMathFunction::nexttoward)
        .value("copysign", sdfg::math::cmath::CMathFunction::copysign)
        // Minimum, maximum, difference functions
        .value("fmax", sdfg::math::cmath::CMathFunction::fmax)
        .value("fmin", sdfg::math::cmath::CMathFunction::fmin)
        .value("fdim", sdfg::math::cmath::CMathFunction::fdim)
        // Other functions
        .value("fma", sdfg::math::cmath::CMathFunction::fma)
        .export_values();
}
