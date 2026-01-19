#include <gtest/gtest.h>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/targets/cuda/plugin.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sdfg::codegen::register_default_dispatchers();
    sdfg::cuda::register_cuda_plugin();
    sdfg::serializer::register_default_serializers();
    return RUN_ALL_TESTS();
}
