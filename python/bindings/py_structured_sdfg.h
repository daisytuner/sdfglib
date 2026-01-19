#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sdfg/structured_sdfg.h>

class PyStructuredSDFGBuilder;

class PyStructuredSDFG {
    friend class PyStructuredSDFGBuilder;

private:
    std::unique_ptr<sdfg::StructuredSDFG> sdfg_;

    PyStructuredSDFG(std::unique_ptr<sdfg::StructuredSDFG>& sdfg);

public:
    static PyStructuredSDFG from_json(const std::string& json_path);

    std::string name() const;

    sdfg::StructuredSDFG& sdfg() { return *sdfg_; }

    const sdfg::types::IType& return_type() const;

    const sdfg::types::IType& type(const std::string& name) const;

    bool exists(const std::string& name) const;

    bool is_argument(const std::string& name) const;

    bool is_transient(const std::string& name) const;

    std::vector<std::string> arguments() const;

    pybind11::dict containers() const;

    void validate();

    void expand();

    void simplify();

    void dump(const std::string& path);

    void normalize();

    void schedule(const std::string& target, const std::string& category);

    std::string compile(
        const std::string& output_folder, const std::string& instrumentation_mode = "", bool capture_args = false
    ) const;

    std::string metadata(const std::string& key) const;
};
