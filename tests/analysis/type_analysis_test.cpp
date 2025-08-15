#include "sdfg/analysis/type_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(TypeAnalysisTest, ScalarType) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    builder.add_container("i", scalar_type);
    builder.add_container("N", scalar_type);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    // 3 * i + 2
    auto update = symbolic::add(symbolic::mul(symbolic::integer(3), indvar), symbolic::integer(2));
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& type_analysis = manager.get<analysis::TypeAnalysis>();

    EXPECT_TRUE(type_analysis.get_outer_type("i") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("i") == types::Scalar(types::PrimitiveType::Int32));
}

TEST(TypeAnalysisTest, PointerType) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Pointer ptr(scalar_type);

    builder.add_container("i", ptr);
    builder.add_container("N", ptr);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    // 3 * i + 2
    auto update = symbolic::add(symbolic::mul(symbolic::integer(3), indvar), symbolic::integer(2));
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& type_analysis = manager.get<analysis::TypeAnalysis>();

    EXPECT_TRUE(type_analysis.get_outer_type("i") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("i") == ptr);
}

TEST(TypeAnalysisTest, ArrayType) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Array arr(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10));

    builder.add_container("i", arr);
    builder.add_container("N", arr);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("i");
    // 3 * i + 2
    auto update = symbolic::add(symbolic::mul(symbolic::integer(3), indvar), symbolic::integer(2));
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& type_analysis = manager.get<analysis::TypeAnalysis>();

    EXPECT_TRUE(type_analysis.get_outer_type("i") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("i") == arr);
}

TEST(TypeAnalysisTest, StructureType) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Structure struct_type("MyStruct");

    builder.add_container("my_struct", struct_type);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto indvar = symbolic::symbol("my_struct");
    // 3 * a + 2
    auto update = symbolic::add(symbolic::mul(symbolic::integer(3), symbolic::symbol("a")), symbolic::integer(2));
    auto condition = symbolic::Lt(symbolic::symbol("a"), symbolic::symbol("b"));
    auto init = symbolic::zero();
    auto& loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager manager(sdfg);
    auto& type_analysis = manager.get<analysis::TypeAnalysis>();

    EXPECT_TRUE(type_analysis.get_outer_type("my_struct") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("my_struct") == struct_type);
}

TEST(TypeAnalysisTest, OpaquePointerTypeRead) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Pointer ptr;
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer array1dType(base_desc);
    types::IType& wrapper = array1dType;
    types::Pointer array2dType(wrapper);

    builder.add_container("i", ptr);
    builder.add_container("N", ptr);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "i");
    auto& access_node2 = builder.add_access(block, "N");

    auto& memlet = builder.add_dereference_memlet(block, access_node, access_node2, true, array2dType);

    analysis::AnalysisManager manager(sdfg);
    auto& type_analysis = manager.get<analysis::TypeAnalysis>();

    EXPECT_TRUE(type_analysis.get_outer_type("i") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("i") == array2dType);

    EXPECT_TRUE(type_analysis.get_outer_type("N") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("N") == array1dType);
}

TEST(TypeAnalysisTest, OpaquePointerTypeWrite) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Pointer ptr;
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer array1dType(base_desc);
    types::IType& wrapper = array1dType;
    types::Pointer array2dType(wrapper);

    builder.add_container("i", ptr);
    builder.add_container("N", ptr);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "i");
    auto& access_node2 = builder.add_access(block, "N");

    auto& memlet = builder.add_dereference_memlet(block, access_node2, access_node, false, array2dType);

    analysis::AnalysisManager manager(sdfg);
    auto& type_analysis = manager.get<analysis::TypeAnalysis>();

    EXPECT_TRUE(type_analysis.get_outer_type("i") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("i") == array2dType);

    EXPECT_TRUE(type_analysis.get_outer_type("N") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("N") == array1dType);
}

TEST(TypeAnalysisTest, OpaquePointerTypeView) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Pointer ptr;
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer array1dType(base_desc);
    types::IType& wrapper = array1dType;
    types::Pointer array2dType(wrapper);

    builder.add_container("i", ptr);
    builder.add_container("N", ptr);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "i");
    auto& access_node2 = builder.add_access(block, "N");

    auto& memlet = builder.add_reference_memlet(block, access_node, access_node2, {symbolic::one()}, array2dType);

    analysis::AnalysisManager manager(sdfg);
    auto& type_analysis = manager.get<analysis::TypeAnalysis>();

    EXPECT_TRUE(type_analysis.get_outer_type("i") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("i") == array2dType);

    EXPECT_TRUE(type_analysis.get_outer_type("N") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("N") == array2dType);
}

TEST(TypeAnalysisTest, TaskletReadWrite) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Pointer ptr;
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer array1dType(base_desc);
    types::IType& wrapper = array1dType;

    builder.add_container("i", ptr);
    builder.add_container("N", base_desc);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "i");
    auto& access_node2 = builder.add_access(block, "N");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});

    auto& read_memlet =
        builder.add_computational_memlet(block, access_node, tasklet, "_in", {symbolic::zero()}, array1dType);
    auto& write_memlet = builder.add_computational_memlet(block, tasklet, "_out", access_node2, {}, base_desc);

    analysis::AnalysisManager manager(sdfg);
    auto& type_analysis = manager.get<analysis::TypeAnalysis>();

    EXPECT_TRUE(type_analysis.get_outer_type("i") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("i") == array1dType);

    EXPECT_TRUE(type_analysis.get_outer_type("N") != nullptr);
    EXPECT_TRUE(*type_analysis.get_outer_type("N") == base_desc);
}
