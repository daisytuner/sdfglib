#include "sdfg/passes/structured_control_flow/for2map.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(For2MapTest, Simple) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Scalar desc_element2(types::PrimitiveType::Int32);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);
    builder.add_container("B", desc_element);
    builder.add_container("i", desc_element2);

    auto& loop = builder.add_for(builder.subject().root(), symbolic::symbol("i"),
                                 symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
                                 symbolic::integer(0),
                                 symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));

    auto& block = builder.add_block(loop.root());

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", desc_element}, {{"_in", desc_element}});
    auto& input_node = builder.add_access(block, "B");
    auto& output_node = builder.add_access(block, "A");
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::symbol("i")});

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2Map conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Map*>(&sdfg->root().at(0).first) !=
                nullptr);
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg->root().at(0).first);
    EXPECT_TRUE(symbolic::eq(map->num_iterations(), symbolic::integer(10)));
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));

    EXPECT_EQ(map->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&map->root().at(0).first) !=
                nullptr);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&map->root().at(0).first);
    EXPECT_EQ(block2->dataflow().nodes().size(), 3);
    EXPECT_EQ(block2->dataflow().edges().size(), 2);

    bool found_input = false;
    bool found_output = false;
    bool found_tasklet = false;
    for (const auto& node : block2->dataflow().nodes()) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                found_input = true;
            } else if (access_node->data() == "B") {
                found_output = true;
            }
        } else if (auto tasklet_node = dynamic_cast<const data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::assign);
            found_tasklet = true;
        }
    }
    EXPECT_TRUE(found_input);
    EXPECT_TRUE(found_output);
    EXPECT_TRUE(found_tasklet);

    // Check the memlets
    bool found_memlet_input = false;
    bool found_memlet_output = false;
    for (const auto& edge : block2->dataflow().edges()) {
        if (edge.src_conn() == "void" && edge.dst_conn() == "_in") {
            found_memlet_input = true;
            EXPECT_TRUE(dynamic_cast<const data_flow::AccessNode*>(&edge.src()) != nullptr);
            auto access_node = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
            EXPECT_EQ(access_node->data(), "B");
            EXPECT_TRUE(dynamic_cast<const data_flow::Tasklet*>(&edge.dst()) != nullptr);
        } else if (edge.src_conn() == "_out" && edge.dst_conn() == "void") {
            found_memlet_output = true;
            EXPECT_TRUE(dynamic_cast<const data_flow::AccessNode*>(&edge.dst()) != nullptr);
            auto access_node = dynamic_cast<const data_flow::AccessNode*>(&edge.dst());
            EXPECT_EQ(access_node->data(), "A");
            EXPECT_TRUE(dynamic_cast<const data_flow::Tasklet*>(&edge.src()) != nullptr);
            EXPECT_EQ(edge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(edge.subset().at(0), symbolic::symbol("i")));
        }
    }
    EXPECT_TRUE(found_memlet_input);
    EXPECT_TRUE(found_memlet_output);
}

TEST(For2MapTest, Strided) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Scalar desc_element2(types::PrimitiveType::Int32);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);
    builder.add_container("B", desc_element);
    builder.add_container("i", desc_element2);
    builder.add_container("n", desc_element2);

    auto& loop = builder.add_for(builder.subject().root(), symbolic::symbol("i"),
                                 symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
                                 symbolic::integer(0),
                                 symbolic::add(symbolic::symbol("i"), symbolic::symbol("n")));

    auto& block = builder.add_block(loop.root());

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", desc_element}, {{"_in", desc_element}});
    auto& input_node = builder.add_access(block, "B");
    auto& output_node = builder.add_access(block, "A");
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::symbol("i")});

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2Map conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Map*>(&sdfg->root().at(0).first) !=
                nullptr);
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg->root().at(0).first);
    EXPECT_TRUE(
        symbolic::eq(map->num_iterations(),
                     symbolic::ceil(symbolic::div(symbolic::integer(10), symbolic::symbol("n")))));
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));

    EXPECT_EQ(map->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&map->root().at(0).first) !=
                nullptr);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&map->root().at(0).first);
    EXPECT_EQ(block2->dataflow().nodes().size(), 3);
    EXPECT_EQ(block2->dataflow().edges().size(), 2);

    bool found_input = false;
    bool found_output = false;
    bool found_tasklet = false;
    for (const auto& node : block2->dataflow().nodes()) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                found_input = true;
            } else if (access_node->data() == "B") {
                found_output = true;
            }
        } else if (auto tasklet_node = dynamic_cast<const data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::assign);
            found_tasklet = true;
        }
    }
    EXPECT_TRUE(found_input);
    EXPECT_TRUE(found_output);
    EXPECT_TRUE(found_tasklet);

    // Check the memlets
    bool found_memlet_input = false;
    bool found_memlet_output = false;
    for (const auto& edge : block2->dataflow().edges()) {
        if (edge.src_conn() == "void" && edge.dst_conn() == "_in") {
            found_memlet_input = true;
            EXPECT_TRUE(dynamic_cast<const data_flow::AccessNode*>(&edge.src()) != nullptr);
            auto access_node = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
            EXPECT_EQ(access_node->data(), "B");
            EXPECT_TRUE(dynamic_cast<const data_flow::Tasklet*>(&edge.dst()) != nullptr);
        } else if (edge.src_conn() == "_out" && edge.dst_conn() == "void") {
            found_memlet_output = true;
            EXPECT_TRUE(dynamic_cast<const data_flow::AccessNode*>(&edge.dst()) != nullptr);
            auto access_node = dynamic_cast<const data_flow::AccessNode*>(&edge.dst());
            EXPECT_EQ(access_node->data(), "A");
            EXPECT_TRUE(dynamic_cast<const data_flow::Tasklet*>(&edge.src()) != nullptr);
            EXPECT_EQ(edge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(edge.subset().at(0),
                                     symbolic::mul(symbolic::symbol("n"), symbolic::symbol("i"))));
        }
    }
    EXPECT_TRUE(found_memlet_input);
    EXPECT_TRUE(found_memlet_output);
}

TEST(For2MapTest, Init) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Scalar desc_element2(types::PrimitiveType::Int32);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);
    builder.add_container("B", desc_element);
    builder.add_container("i", desc_element2);

    auto& loop = builder.add_for(builder.subject().root(), symbolic::symbol("i"),
                                 symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
                                 symbolic::integer(2),
                                 symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));

    auto& block = builder.add_block(loop.root());

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", desc_element}, {{"_in", desc_element}});
    auto& input_node = builder.add_access(block, "B");
    auto& output_node = builder.add_access(block, "A");
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::symbol("i")});

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2Map conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Map*>(&sdfg->root().at(0).first) !=
                nullptr);
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg->root().at(0).first);
    EXPECT_TRUE(symbolic::eq(map->num_iterations(), symbolic::integer(8)));
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));

    EXPECT_EQ(map->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&map->root().at(0).first) !=
                nullptr);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&map->root().at(0).first);
    EXPECT_EQ(block2->dataflow().nodes().size(), 3);
    EXPECT_EQ(block2->dataflow().edges().size(), 2);

    bool found_input = false;
    bool found_output = false;
    bool found_tasklet = false;
    for (const auto& node : block2->dataflow().nodes()) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                found_input = true;
            } else if (access_node->data() == "B") {
                found_output = true;
            }
        } else if (auto tasklet_node = dynamic_cast<const data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::assign);
            found_tasklet = true;
        }
    }
    EXPECT_TRUE(found_input);
    EXPECT_TRUE(found_output);
    EXPECT_TRUE(found_tasklet);

    // Check the memlets
    bool found_memlet_input = false;
    bool found_memlet_output = false;
    for (const auto& edge : block2->dataflow().edges()) {
        if (edge.src_conn() == "void" && edge.dst_conn() == "_in") {
            found_memlet_input = true;
            EXPECT_TRUE(dynamic_cast<const data_flow::AccessNode*>(&edge.src()) != nullptr);
            auto access_node = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
            EXPECT_EQ(access_node->data(), "B");
            EXPECT_TRUE(dynamic_cast<const data_flow::Tasklet*>(&edge.dst()) != nullptr);
        } else if (edge.src_conn() == "_out" && edge.dst_conn() == "void") {
            found_memlet_output = true;
            EXPECT_TRUE(dynamic_cast<const data_flow::AccessNode*>(&edge.dst()) != nullptr);
            auto access_node = dynamic_cast<const data_flow::AccessNode*>(&edge.dst());
            EXPECT_EQ(access_node->data(), "A");
            EXPECT_TRUE(dynamic_cast<const data_flow::Tasklet*>(&edge.src()) != nullptr);
            EXPECT_EQ(edge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(edge.subset().at(0),
                                     symbolic::add(symbolic::integer(2), symbolic::symbol("i"))));
        }
    }
    EXPECT_TRUE(found_memlet_input);
    EXPECT_TRUE(found_memlet_output);
}

TEST(For2MapTest, Bound) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Scalar desc_element2(types::PrimitiveType::Int32);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);
    builder.add_container("B", desc_element);
    builder.add_container("i", desc_element2);
    builder.add_container("n", desc_element2);

    auto& loop = builder.add_for(
        builder.subject().root(), symbolic::symbol("i"),
        symbolic::Lt(symbolic::add(symbolic::symbol("i"), symbolic::symbol("n")),
                     symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));

    auto& block = builder.add_block(loop.root());

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", desc_element}, {{"_in", desc_element}});
    auto& input_node = builder.add_access(block, "B");
    auto& output_node = builder.add_access(block, "A");
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::symbol("i")});

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2Map conversion_pass;
    EXPECT_TRUE(conversion_pass.run(builder_opt, analysis_manager));

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Map*>(&sdfg->root().at(0).first) !=
                nullptr);
    auto map = dynamic_cast<const structured_control_flow::Map*>(&sdfg->root().at(0).first);
    EXPECT_TRUE(symbolic::eq(map->num_iterations(),
                             symbolic::sub(symbolic::integer(10), symbolic::symbol("n"))));
    EXPECT_TRUE(symbolic::eq(map->indvar(), symbolic::symbol("i")));

    EXPECT_EQ(map->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&map->root().at(0).first) !=
                nullptr);
    auto block2 = dynamic_cast<const structured_control_flow::Block*>(&map->root().at(0).first);
    EXPECT_EQ(block2->dataflow().nodes().size(), 3);
    EXPECT_EQ(block2->dataflow().edges().size(), 2);

    bool found_input = false;
    bool found_output = false;
    bool found_tasklet = false;
    for (const auto& node : block2->dataflow().nodes()) {
        if (auto access_node = dynamic_cast<const data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                found_input = true;
            } else if (access_node->data() == "B") {
                found_output = true;
            }
        } else if (auto tasklet_node = dynamic_cast<const data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::assign);
            found_tasklet = true;
        }
    }
    EXPECT_TRUE(found_input);
    EXPECT_TRUE(found_output);
    EXPECT_TRUE(found_tasklet);

    // Check the memlets
    bool found_memlet_input = false;
    bool found_memlet_output = false;
    for (const auto& edge : block2->dataflow().edges()) {
        if (edge.src_conn() == "void" && edge.dst_conn() == "_in") {
            found_memlet_input = true;
            EXPECT_TRUE(dynamic_cast<const data_flow::AccessNode*>(&edge.src()) != nullptr);
            auto access_node = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
            EXPECT_EQ(access_node->data(), "B");
            EXPECT_TRUE(dynamic_cast<const data_flow::Tasklet*>(&edge.dst()) != nullptr);
        } else if (edge.src_conn() == "_out" && edge.dst_conn() == "void") {
            found_memlet_output = true;
            EXPECT_TRUE(dynamic_cast<const data_flow::AccessNode*>(&edge.dst()) != nullptr);
            auto access_node = dynamic_cast<const data_flow::AccessNode*>(&edge.dst());
            EXPECT_EQ(access_node->data(), "A");
            EXPECT_TRUE(dynamic_cast<const data_flow::Tasklet*>(&edge.src()) != nullptr);
            EXPECT_EQ(edge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(edge.subset().at(0), symbolic::symbol("i")));
        }
    }
    EXPECT_TRUE(found_memlet_input);
    EXPECT_TRUE(found_memlet_output);
}

TEST(For2MapTest, Failed_update) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Scalar desc_element2(types::PrimitiveType::Int32);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);
    builder.add_container("B", desc_element);
    builder.add_container("i", desc_element2);

    auto& loop = builder.add_for(builder.subject().root(), symbolic::symbol("i"),
                                 symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
                                 symbolic::integer(0),
                                 symbolic::mul(symbolic::symbol("i"), symbolic::integer(2)));

    auto& block = builder.add_block(loop.root());

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", desc_element}, {{"_in", desc_element}});
    auto& input_node = builder.add_access(block, "B");
    auto& output_node = builder.add_access(block, "A");
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::symbol("i")});

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2Map conversion_pass;
    EXPECT_FALSE(conversion_pass.run(builder_opt, analysis_manager));
}

TEST(For2MapTest, Failed_bound) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    types::Scalar desc_element(types::PrimitiveType::Double);
    types::Scalar desc_element2(types::PrimitiveType::Int32);
    types::Array desc_array(desc_element, symbolic::integer(10));
    builder.add_container("A", desc_array);
    builder.add_container("B", desc_element);
    builder.add_container("i", desc_element2);

    auto& loop = builder.add_for(
        builder.subject().root(), symbolic::symbol("i"),
        symbolic::Lt(symbolic::mul(symbolic::symbol("i"), symbolic::integer(2)),
                     symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(2)));

    auto& block = builder.add_block(loop.root());

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", desc_element}, {{"_in", desc_element}});
    auto& input_node = builder.add_access(block, "B");
    auto& output_node = builder.add_access(block, "A");
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::symbol("i")});

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    // Fusion
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::For2Map conversion_pass;
    EXPECT_FALSE(conversion_pass.run(builder_opt, analysis_manager));
}
