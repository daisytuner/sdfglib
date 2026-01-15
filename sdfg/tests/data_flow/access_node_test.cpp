#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
using namespace sdfg;

TEST(AccessNodeTest, BasicCreation) {
    builder::SDFGBuilder builder("sdfg_access_node_basic", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("data", desc);

    auto& access_node = builder.add_access(state, "data");

    EXPECT_EQ(access_node.data(), "data");
}

TEST(AccessNodeTest, DataGetter) {
    builder::SDFGBuilder builder("sdfg_access_node_data", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("variable", desc);

    auto& access_node = builder.add_access(state, "variable");

    EXPECT_EQ(access_node.data(), "variable");
}

TEST(AccessNodeTest, DataSetter) {
    builder::SDFGBuilder builder("sdfg_access_node_setter", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("old_name", desc);
    builder.add_container("new_name", desc);

    auto& access_node = builder.add_access(state, "old_name");
    EXPECT_EQ(access_node.data(), "old_name");

    access_node.data("new_name");
    EXPECT_EQ(access_node.data(), "new_name");
}

TEST(AccessNodeTest, MultipleAccessNodes) {
    builder::SDFGBuilder builder("sdfg_multi_access", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc1(types::PrimitiveType::Float);
    types::Scalar desc2(types::PrimitiveType::Int32);
    builder.add_container("data1", desc1);
    builder.add_container("data2", desc2);

    auto& access_node_1 = builder.add_access(state, "data1");
    auto& access_node_2 = builder.add_access(state, "data2");

    EXPECT_EQ(access_node_1.data(), "data1");
    EXPECT_EQ(access_node_2.data(), "data2");
    EXPECT_NE(&access_node_1, &access_node_2);
}

TEST(AccessNodeTest, CloneNode) {
    builder::SDFGBuilder builder("sdfg_clone_access", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt64);
    builder.add_container("original", desc);

    auto& access_node = builder.add_access(state, "original");
    EXPECT_EQ(access_node.data(), "original");
}

TEST(AccessNodeTest, ReplaceSymbol) {
    builder::SDFGBuilder builder("sdfg_replace_symbol", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("old_symbol", desc);
    builder.add_container("new_symbol", desc);

    auto& access_node = builder.add_access(state, "old_symbol");
    EXPECT_EQ(access_node.data(), "old_symbol");

    auto old_expr = symbolic::symbol("old_symbol");
    auto new_expr = symbolic::symbol("new_symbol");
    access_node.replace(old_expr, new_expr);

    EXPECT_EQ(access_node.data(), "new_symbol");
}

TEST(AccessNodeTest, ParentGraph) {
    builder::SDFGBuilder builder("sdfg_parent", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Bool);
    builder.add_container("data", desc);

    auto& access_node = builder.add_access(state, "data");
    auto& parent_graph = access_node.get_parent();

    EXPECT_EQ(&parent_graph, &state.dataflow());
}

TEST(ConstantNodeTest, BasicCreation) {
    builder::SDFGBuilder builder("sdfg_const_basic", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    auto& constant_node = builder.add_constant(state, "42", desc);

    EXPECT_EQ(constant_node.data(), "42");
}

TEST(ConstantNodeTest, IntegerConstant) {
    builder::SDFGBuilder builder("sdfg_const_int", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int64);
    auto& constant_node = builder.add_constant(state, "12345", desc);

    EXPECT_EQ(constant_node.data(), "12345");
    EXPECT_EQ(constant_node.type().type_id(), types::TypeID::Scalar);
}

TEST(ConstantNodeTest, FloatConstant) {
    builder::SDFGBuilder builder("sdfg_const_float", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    auto& constant_node = builder.add_constant(state, "3.14", desc);

    EXPECT_EQ(constant_node.data(), "3.14");
}

TEST(ConstantNodeTest, BoolConstantTrue) {
    builder::SDFGBuilder builder("sdfg_const_bool_true", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Bool);
    auto& constant_node = builder.add_constant(state, "true", desc);

    EXPECT_EQ(constant_node.data(), "true");
}

TEST(ConstantNodeTest, BoolConstantFalse) {
    builder::SDFGBuilder builder("sdfg_const_bool_false", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Bool);
    auto& constant_node = builder.add_constant(state, "false", desc);

    EXPECT_EQ(constant_node.data(), "false");
}

TEST(ConstantNodeTest, NegativeIntegerConstant) {
    builder::SDFGBuilder builder("sdfg_const_neg", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    auto& constant_node = builder.add_constant(state, "-100", desc);

    EXPECT_EQ(constant_node.data(), "-100");
}

TEST(ConstantNodeTest, ZeroConstant) {
    builder::SDFGBuilder builder("sdfg_const_zero", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    auto& constant_node = builder.add_constant(state, "0", desc);

    EXPECT_EQ(constant_node.data(), "0");
}

TEST(ConstantNodeTest, TypeGetter) {
    builder::SDFGBuilder builder("sdfg_const_type", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    auto& constant_node = builder.add_constant(state, "2.718", desc);

    EXPECT_EQ(constant_node.type().type_id(), types::TypeID::Scalar);
    auto& scalar_type = static_cast<const types::Scalar&>(constant_node.type());
    EXPECT_EQ(scalar_type.primitive_type(), types::PrimitiveType::Double);
}

TEST(ConstantNodeTest, MultipleConstants) {
    builder::SDFGBuilder builder("sdfg_multi_const", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Scalar float_type(types::PrimitiveType::Float);

    auto& const1 = builder.add_constant(state, "10", int_type);
    auto& const2 = builder.add_constant(state, "3.14", float_type);

    EXPECT_EQ(const1.data(), "10");
    EXPECT_EQ(const2.data(), "3.14");
    EXPECT_NE(&const1, &const2);
}

TEST(ConstantNodeTest, ConstantToTasklet) {
    builder::SDFGBuilder builder("sdfg_const_tasklet", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("output", desc);

    auto& constant_node = builder.add_constant(state, "42", desc);
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& output = builder.add_access(state, "output");

    builder.add_computational_memlet(state, constant_node, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    EXPECT_EQ(constant_node.data(), "42");
}

TEST(ConstantNodeTest, UInt8Constant) {
    builder::SDFGBuilder builder("sdfg_const_uint8", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt8);
    auto& constant_node = builder.add_constant(state, "255", desc);

    EXPECT_EQ(constant_node.data(), "255");
}

TEST(ConstantNodeTest, UInt64Constant) {
    builder::SDFGBuilder builder("sdfg_const_uint64", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt64);
    auto& constant_node = builder.add_constant(state, "18446744073709551615", desc);

    EXPECT_EQ(constant_node.data(), "18446744073709551615");
}
