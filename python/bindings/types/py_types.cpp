#include "py_types.h"

#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/array.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/structure.h>
#include <sdfg/types/type.h>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace sdfg::types;

void register_types(py::module& m) {
    // PrimitiveType
    py::enum_<PrimitiveType>(m, "PrimitiveType")
        .value("Void", PrimitiveType::Void)
        .value("Bool", PrimitiveType::Bool)
        .value("Int8", PrimitiveType::Int8)
        .value("Int16", PrimitiveType::Int16)
        .value("Int32", PrimitiveType::Int32)
        .value("Int64", PrimitiveType::Int64)
        .value("Int128", PrimitiveType::Int128)
        .value("UInt8", PrimitiveType::UInt8)
        .value("UInt16", PrimitiveType::UInt16)
        .value("UInt32", PrimitiveType::UInt32)
        .value("UInt64", PrimitiveType::UInt64)
        .value("UInt128", PrimitiveType::UInt128)
        .value("Half", PrimitiveType::Half)
        .value("BFloat", PrimitiveType::BFloat)
        .value("Float", PrimitiveType::Float)
        .value("Double", PrimitiveType::Double)
        .value("X86_FP80", PrimitiveType::X86_FP80)
        .value("FP128", PrimitiveType::FP128)
        .value("PPC_FP128", PrimitiveType::PPC_FP128)
        .export_values();

    // IType
    py::class_<IType>(m, "Type")
        .def("print", &IType::print)
        .def("__repr__", &IType::print)
        .def_property_readonly("primitive_type", &IType::primitive_type);

    // Scalar
    py::class_<Scalar, IType>(m, "Scalar").def(py::init<PrimitiveType>(), py::arg("primitive_type"));

    // Array
    py::class_<Array, IType>(m, "Array")
        .def(
            py::init([](const IType& element_type, const std::string& num_elements) {
                return new Array(element_type, sdfg::symbolic::parse(num_elements));
            }),
            py::arg("element_type"),
            py::arg("num_elements")
        )
        .def_property_readonly("element_type", &Array::element_type)
        .def_property_readonly("num_elements", [](const Array& self) { return self.num_elements()->__str__(); });

    // Pointer
    py::class_<Pointer, IType>(m, "Pointer")
        .def(py::init<>())
        .def(py::init<const IType&>(), py::arg("pointee_type"))
        .def_property_readonly("pointee_type", &Pointer::pointee_type)
        .def("has_pointee_type", &Pointer::has_pointee_type);

    // Structure
    py::class_<Structure, IType>(m, "Structure")
        .def(py::init<const std::string&>(), py::arg("name"))
        .def_property_readonly("name", &Structure::name);
}
