#include <gtest/gtest.h>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/rpc/rpc_loop_opt.h"
#include "sdfg/serializer/json_serializer.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();

    sdfg::passes::rpc::SimpleRpcContextBuilder b;
    b.initialize_local_default();
    b.from_env();
    b.from_docc_config();
    b.server = "http://localhost:8080/docc";
    auto ctx = b.build();

    sdfg::passes::rpc::register_rpc_loop_opt(*ctx, "sequential", "server", true);

    return RUN_ALL_TESTS();
}
