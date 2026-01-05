#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
using namespace sdfg;

TEST(CodeNodeTest, TaskletInputsOutputs) {
    builder::SDFGBuilder builder("sdfg_code_node_io", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("in1", desc);
    builder.add_container("out1", desc);

    auto& input = builder.add_access(state, "in1");
    auto& output = builder.add_access(state, "out1");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, input, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    EXPECT_EQ(tasklet.outputs().size(), 1);
    EXPECT_EQ(tasklet.inputs().size(), 1);
    EXPECT_EQ(tasklet.outputs()[0], "_out");
    EXPECT_EQ(tasklet.inputs()[0], "_in");
}

TEST(CodeNodeTest, MultipleInputs) {
    builder::SDFGBuilder builder("sdfg_multi_inputs", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("result", desc);

    auto& a_node = builder.add_access(state, "a");
    auto& b_node = builder.add_access(state, "b");
    auto& result_node = builder.add_access(state, "result");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

    builder.add_computational_memlet(state, a_node, tasklet, "_in1", {});
    builder.add_computational_memlet(state, b_node, tasklet, "_in2", {});
    builder.add_computational_memlet(state, tasklet, "_out", result_node, {});

    EXPECT_EQ(tasklet.inputs().size(), 2);
    EXPECT_EQ(tasklet.inputs()[0], "_in1");
    EXPECT_EQ(tasklet.inputs()[1], "_in2");
}

TEST(CodeNodeTest, MultipleOutputs) {
    builder::SDFGBuilder builder("sdfg_multi_outputs", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("input", desc);
    builder.add_container("out1", desc);
    builder.add_container("out2", desc);

    auto& input = builder.add_access(state, "input");
    auto& output1 = builder.add_access(state, "out1");
    auto& output2 = builder.add_access(state, "out2");

    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out1", {"_in"});
    tasklet.outputs().push_back("_out2");

    builder.add_computational_memlet(state, input, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out1", output1, {});
    builder.add_computational_memlet(state, tasklet, "_out2", output2, {});

    EXPECT_EQ(tasklet.outputs().size(), 2);
    EXPECT_EQ(tasklet.outputs()[0], "_out1");
    EXPECT_EQ(tasklet.outputs()[1], "_out2");
}

TEST(CodeNodeTest, OutputByIndex) {
    builder::SDFGBuilder builder("sdfg_output_index", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("in", desc);
    builder.add_container("out", desc);

    auto& input = builder.add_access(state, "in");
    auto& output = builder.add_access(state, "out");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_output", {"_input"});

    builder.add_computational_memlet(state, input, tasklet, "_input", {});
    builder.add_computational_memlet(state, tasklet, "_output", output, {});

    EXPECT_EQ(tasklet.output(0), "_output");
}

TEST(CodeNodeTest, InputByIndex) {
    builder::SDFGBuilder builder("sdfg_input_index", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("x", desc);
    builder.add_container("y", desc);
    builder.add_container("z", desc);

    auto& x_node = builder.add_access(state, "x");
    auto& y_node = builder.add_access(state, "y");
    auto& z_node = builder.add_access(state, "z");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});

    builder.add_computational_memlet(state, x_node, tasklet, "_in1", {});
    builder.add_computational_memlet(state, y_node, tasklet, "_in2", {});
    builder.add_computational_memlet(state, tasklet, "_out", z_node, {});

    EXPECT_EQ(tasklet.input(0), "_in1");
    EXPECT_EQ(tasklet.input(1), "_in2");
}

TEST(CodeNodeTest, HasConstantInput) {
    builder::SDFGBuilder builder("sdfg_const_input", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("output", desc);

    auto& constant = builder.add_constant(state, "10", desc);
    auto& output = builder.add_access(state, "output");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, constant, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    EXPECT_TRUE(tasklet.has_constant_input(0));
}

TEST(CodeNodeTest, HasNoConstantInput) {
    builder::SDFGBuilder builder("sdfg_no_const_input", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("input", desc);
    builder.add_container("output", desc);

    auto& input = builder.add_access(state, "input");
    auto& output = builder.add_access(state, "output");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, input, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    EXPECT_FALSE(tasklet.has_constant_input(0));
}

TEST(CodeNodeTest, MixedConstantAndRegularInputs) {
    builder::SDFGBuilder builder("sdfg_mixed_inputs", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("var", desc);
    builder.add_container("result", desc);

    auto& constant = builder.add_constant(state, "5.0", desc);
    auto& var_node = builder.add_access(state, "var");
    auto& result_node = builder.add_access(state, "result");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

    builder.add_computational_memlet(state, constant, tasklet, "_in1", {});
    builder.add_computational_memlet(state, var_node, tasklet, "_in2", {});
    builder.add_computational_memlet(state, tasklet, "_out", result_node, {});

    EXPECT_TRUE(tasklet.has_constant_input(0));
    EXPECT_FALSE(tasklet.has_constant_input(1));
}

TEST(CodeNodeTest, ThreeInputs) {
    builder::SDFGBuilder builder("sdfg_three_inputs", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("result", desc);

    auto& a_node = builder.add_access(state, "a");
    auto& b_node = builder.add_access(state, "b");
    auto& c_node = builder.add_access(state, "c");
    auto& result_node = builder.add_access(state, "result");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});

    builder.add_computational_memlet(state, a_node, tasklet, "_in1", {});
    builder.add_computational_memlet(state, b_node, tasklet, "_in2", {});
    builder.add_computational_memlet(state, c_node, tasklet, "_in3", {});
    builder.add_computational_memlet(state, tasklet, "_out", result_node, {});

    EXPECT_EQ(tasklet.inputs().size(), 3);
    EXPECT_EQ(tasklet.input(0), "_in1");
    EXPECT_EQ(tasklet.input(1), "_in2");
    EXPECT_EQ(tasklet.input(2), "_in3");
}

TEST(CodeNodeTest, GetParentGraph) {
    builder::SDFGBuilder builder("sdfg_parent_graph", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("in", desc);
    builder.add_container("out", desc);

    auto& input = builder.add_access(state, "in");
    auto& output = builder.add_access(state, "out");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, input, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    auto& parent = tasklet.get_parent();
    EXPECT_EQ(&parent, &state.dataflow());
}

TEST(CodeNodeTest, IntegerOperations) {
    builder::SDFGBuilder builder("sdfg_int_ops", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("x", desc);
    builder.add_container("y", desc);
    builder.add_container("z", desc);

    auto& x_node = builder.add_access(state, "x");
    auto& y_node = builder.add_access(state, "y");
    auto& z_node = builder.add_access(state, "z");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::int_add, "_out", {"_in1", "_in2"});

    builder.add_computational_memlet(state, x_node, tasklet, "_in1", {});
    builder.add_computational_memlet(state, y_node, tasklet, "_in2", {});
    builder.add_computational_memlet(state, tasklet, "_out", z_node, {});

    EXPECT_EQ(tasklet.inputs().size(), 2);
    EXPECT_EQ(tasklet.outputs().size(), 1);
}

TEST(CodeNodeTest, ModifyInputsList) {
    builder::SDFGBuilder builder("sdfg_modify_inputs", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    builder.add_container("in", desc);
    builder.add_container("out", desc);

    auto& input = builder.add_access(state, "in");
    auto& output = builder.add_access(state, "out");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, input, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    auto& inputs = tasklet.inputs();
    EXPECT_EQ(inputs.size(), 1);
    EXPECT_EQ(inputs[0], "_in");
}

TEST(CodeNodeTest, ModifyOutputsList) {
    builder::SDFGBuilder builder("sdfg_modify_outputs", FunctionType_CPU);
    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("in", desc);
    builder.add_container("out", desc);

    auto& input = builder.add_access(state, "in");
    auto& output = builder.add_access(state, "out");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});

    builder.add_computational_memlet(state, input, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", output, {});

    auto& outputs = tasklet.outputs();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0], "_out");
}
