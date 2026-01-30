#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_generators/c_code_generator.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/instrumentation/arg_capture_plan.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/function.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/types/array.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(test, test) {
    builder::StructuredSDFGBuilder builder("test", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer test_desc(types::StorageType("CPU_Stack", SymEngine::null, types::StorageType::Unmanaged, types::StorageType::Unmanaged), 0, "{1, 2, 3, 4}", base_desc);
    builder.add_container("a", test_desc, false, false);

    auto& sdfg = builder.subject();
    auto instrumentation_plan = codegen::InstrumentationPlan::none(sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(sdfg);
    analysis::AnalysisManager analysis_manager(sdfg);
    codegen::CPPCodeGenerator generator(sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    EXPECT_TRUE(generator.generate());
    std::cout << generator.globals().str() << std::endl;
    std::cout << generator.main().str() << std::endl;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sdfg::codegen::register_default_dispatchers();
    sdfg::serializer::register_default_serializers();
    return RUN_ALL_TESTS();
}
