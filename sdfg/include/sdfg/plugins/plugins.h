#pragma once

#include <list>
#include <memory>
#include <string>

#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace plugins {

struct Plugin {
    const char* name;
    const char* version;
    const char* description;

    // Register callback
    void (*register_plugin_callback)();

    // SDFG lookup
    std::list<std::unique_ptr<sdfg::StructuredSDFG>> (*sdfg_lookup)(std::string name);
};

} // namespace plugins
} // namespace sdfg
