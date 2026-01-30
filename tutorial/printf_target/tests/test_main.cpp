#include <gtest/gtest.h>

#include "plugin.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/serializer/json_serializer.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    // Register default dispatchers and serializers
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();

    // Register the printf target plugin
    sdfg::printf_target::register_printf_plugin();

    return RUN_ALL_TESTS();
}
