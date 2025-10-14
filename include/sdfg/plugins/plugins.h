#pragma once

#include <string>

namespace sdfg {
namespace plugins {

struct Plugin {
    const char* name;
    const char* version;
    const char* description;

    // Register callback
    void (*register_plugin_callback)();
};

}
}
