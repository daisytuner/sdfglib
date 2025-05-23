#include "sdfg/analysis/happens_before_analysis.h"

#include <gtest/gtest.h>

#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(HappensBeforeAnalysisTest, VisitBlock_WAR) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_block(users, block, open_reads, open_reads_after_writes,
                         closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 1);
    EXPECT_EQ(open_reads_after_writes.size(), 1);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    auto read = *open_reads.begin();
    auto write = *open_reads_after_writes.begin();

    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &input_node);
    EXPECT_EQ(write.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write.first->container(), "B");
    EXPECT_EQ(write.first->element(), &output_node);
    EXPECT_EQ(write.second.size(), 0);
}

TEST(HappensBeforeAnalysisTest, VisitBlock_RAW) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& output_node2 = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block, output_node, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_block(users, block, open_reads, open_reads_after_writes,
                         closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 1);
    EXPECT_EQ(open_reads_after_writes.size(), 2);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    auto read = *open_reads.begin();

    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &input_node);

    bool foundB = false;
    bool foundC = false;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);

            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->use(), analysis::Use::READ);
            EXPECT_EQ((*entry.second.begin())->container(), "B");
            EXPECT_EQ((*entry.second.begin())->element(), &output_node);
        } else if (entry.first->container() == "C") {
            foundC = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "C");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 0);
        }
    }

    EXPECT_TRUE(foundB && foundC);
}

TEST(HappensBeforeAnalysisTest, VisitBlock_WAW) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& input_node3 = output_node;
    auto& output_node3 = builder.add_access(block, "C");
    auto& tasklet3 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet3, "_out", output_node3, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node3, "void", tasklet3, "_in", {symbolic::integer(0)});

    auto& output_node2 = builder.add_access(block, "B");
    auto& input_node2 = output_node3;
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_block(users, block, open_reads, open_reads_after_writes,
                         closed_reads_after_write);

    // Check result

    // Open Reads
    EXPECT_EQ(open_reads.size(), 1);

    auto read = *open_reads.begin();
    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &input_node);

    // Closed Writes
    EXPECT_EQ(closed_reads_after_write.size(), 1);

    auto write = *closed_reads_after_write.begin();
    EXPECT_EQ(write.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write.first->container(), "B");
    EXPECT_EQ(write.first->element(), &output_node);
    EXPECT_EQ(write.second.size(), 1);
    auto& raw = *write.second.begin();
    EXPECT_EQ(raw->container(), "B");
    EXPECT_EQ(raw->element(), &input_node3);

    // Open Reads after Writes
    EXPECT_EQ(open_reads_after_writes.size(), 2);

    bool foundB = false;
    bool foundC = false;
    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "C") {
            foundC = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "C");
            EXPECT_EQ(entry.first->element(), &output_node3);
            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->container(), "C");
        }
    }

    EXPECT_TRUE(foundB && foundC);
}

TEST(HappensBeforeAnalysisTest, VisitBlock_SingleMemlet) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"0", types::Scalar(types::PrimitiveType::Int32)}});
    auto& edge =
        builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::symbol("i")});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_block(users, block, open_reads, open_reads_after_writes,
                         closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 1);
    EXPECT_EQ(open_reads_after_writes.size(), 1);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    bool foundA = false;
    auto& read = *open_reads.begin();
    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "i");
    EXPECT_EQ(read->element(), &edge);

    auto& entry = *open_reads_after_writes.begin();
    EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(entry.first->container(), "A");
    EXPECT_EQ(entry.first->element(), &output_node);
    EXPECT_EQ(entry.second.size(), 0);
}

TEST(HappensBeforeAnalysisTest, VisitBlock_MultiMemlet) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::add,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in1", types::Scalar(types::PrimitiveType::Int32)},
                                         {"_in2", types::Scalar(types::PrimitiveType::Int32)}});
    auto& iedge1 =
        builder.add_memlet(block, input_node, "void", tasklet, "_in1", {symbolic::symbol("i")});
    auto& iedge2 =
        builder.add_memlet(block, input_node, "void", tasklet, "_in2", {symbolic::symbol("i")});
    auto& edge1 =
        builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::symbol("i")});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_block(users, block, open_reads, open_reads_after_writes,
                         closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 4);
    EXPECT_EQ(open_reads_after_writes.size(), 1);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    bool foundi1 = false;
    bool foundi2 = false;
    bool foundi3 = false;
    bool foundA = false;
    for (auto read : open_reads) {
        if (read->container() == "i") {
            if (read->element() == &iedge1) {
                foundi1 = true;
                EXPECT_EQ(read->use(), analysis::Use::READ);
                EXPECT_EQ(read->container(), "i");
                EXPECT_EQ(read->element(), &iedge1);
            } else if (read->element() == &iedge2) {
                foundi2 = true;
                EXPECT_EQ(read->use(), analysis::Use::READ);
                EXPECT_EQ(read->container(), "i");
                EXPECT_EQ(read->element(), &iedge2);
            } else if (read->element() == &edge1) {
                foundi3 = true;
                EXPECT_EQ(read->use(), analysis::Use::READ);
                EXPECT_EQ(read->container(), "i");
                EXPECT_EQ(read->element(), &edge1);
            }
        } else if (read->container() == "A") {
            foundA = true;
            EXPECT_EQ(read->use(), analysis::Use::READ);
            EXPECT_EQ(read->container(), "A");
            EXPECT_EQ(read->element(), &input_node);
        }
    }
    EXPECT_TRUE(foundi1 && foundi2 && foundi3 && foundA);

    auto& entry = *open_reads_after_writes.begin();
    EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(entry.first->container(), "A");
    EXPECT_EQ(entry.first->element(), &output_node);
    EXPECT_EQ(entry.second.size(), 0);
}

TEST(HappensBeforeAnalysisTest, visit_for) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& for_loop = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));

    auto& block = builder.add_block(for_loop.root());
    auto& input_node = builder.add_access(block, "i");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_for(users, for_loop, open_reads, open_reads_after_writes,
                       closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 0);
    EXPECT_EQ(open_reads_after_writes.size(), 3);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    bool foundB = false;
    int both_i = 0;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "i") {
            both_i++;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "i");
            EXPECT_EQ(entry.first->element(), &for_loop);
            EXPECT_EQ(entry.second.size(), 3);
        }
    }

    EXPECT_TRUE(foundB);
    EXPECT_EQ(both_i, 2);
}

TEST(HappensBeforeAnalysisTest, visit_map) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& map = builder.add_map(root, symbolic::symbol("i"), symbolic::integer(10));

    auto& block = builder.add_block(map.root());
    auto& input_node = builder.add_access(block, "i");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    std::cout << "Map: " << map.name() << std::endl;

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::cout << "Map: " << map.name() << std::endl;

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_map(users, map, open_reads, open_reads_after_writes, closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 0);
    EXPECT_EQ(open_reads_after_writes.size(), 2);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    bool foundB = false;
    bool foundi = false;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "i") {
            foundi = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "i");
            EXPECT_EQ(entry.first->element(), &map);
            EXPECT_EQ(entry.second.size(), 1);
        }
    }

    EXPECT_TRUE(foundB);
    EXPECT_TRUE(foundi);
}

TEST(HappensBeforeAnalysisTest, visit_while) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& while_loop = builder.add_while(root);

    auto& block = builder.add_block(while_loop.root());
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& output_node2 = builder.add_access(block, "A");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block, output_node, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_while(users, while_loop, open_reads, open_reads_after_writes,
                         closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 1);
    auto& oread = *open_reads.begin();
    EXPECT_EQ(oread->use(), analysis::Use::READ);
    EXPECT_EQ(oread->container(), "A");
    EXPECT_EQ(oread->element(), &input_node);

    EXPECT_EQ(open_reads_after_writes.size(), 2);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    bool foundA = false;
    bool foundB = false;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "A") {
            foundA = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "A");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->element(), &input_node);
        } else if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->element(), &output_node);
        }
    }
}

TEST(HappensBeforeAnalysisTest, visit_if_else_complete) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& if_else = builder.add_if_else(root);
    auto& true_case =
        builder.add_case(if_else, symbolic::Le(symbolic::symbol("A"), symbolic::integer(0)));
    auto& false_case =
        builder.add_case(if_else, symbolic::Gt(symbolic::symbol("A"), symbolic::integer(0)));

    auto& block = builder.add_block(true_case);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& block2 = builder.add_block(false_case);
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_if_else(users, if_else, open_reads, open_reads_after_writes,
                           closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 3);
    EXPECT_EQ(open_reads_after_writes.size(), 2);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    bool foundA = false;
    bool foundB = false;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "A") {
            foundA = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "A");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 0);
        }
    }

    EXPECT_TRUE(foundA && foundB);

    int A_count = 0;
    foundA = false;
    foundB = false;

    for (auto entry : open_reads) {
        if (entry->container() == "A") {
            EXPECT_EQ(entry->use(), analysis::Use::READ);
            EXPECT_EQ(entry->container(), "A");
            if (entry->element() == &input_node) {
                foundA = true;
            } else {
                EXPECT_EQ(entry->element(), &if_else);
                A_count++;
            }
        } else if (entry->container() == "B") {
            EXPECT_EQ(entry->use(), analysis::Use::READ);
            EXPECT_EQ(entry->container(), "B");
            EXPECT_EQ(entry->element(), &input_node2);
            foundB = true;
        }
    }

    EXPECT_EQ(A_count, 1);
    EXPECT_TRUE(foundA && foundB);
}

TEST(HappensBeforeAnalysisTest, visit_kernel) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& kernel = builder.add_kernel(root, "kernel");

    auto& block = builder.add_block(kernel.root());
    auto& input_node = builder.add_access(block, kernel.threadIdx_x()->get_name());
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_kernel(users, kernel, open_reads, open_reads_after_writes,
                          closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 1);
    EXPECT_EQ(open_reads_after_writes.size(), 1);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    if (open_reads.size() == 1) {
        auto read = *open_reads.begin();
        EXPECT_EQ(read->use(), analysis::Use::READ);
        EXPECT_EQ(read->container(), kernel.threadIdx_x()->get_name());
        EXPECT_EQ(read->element(), &input_node);
    }

    bool foundB = false;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 0);
        }
    }

    EXPECT_TRUE(foundB);
}

TEST(HappensBeforeAnalysisTest, visit_sequence_blocks_RAW) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);

    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "C");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_sequence(users, builder_opt.subject().root(), open_reads,
                            open_reads_after_writes, closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 1);
    EXPECT_EQ(open_reads_after_writes.size(), 2);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    auto read = *open_reads.begin();

    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &input_node);

    bool foundB = false;
    bool foundC = false;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->container(), "B");
        } else if (entry.first->container() == "C") {
            foundC = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "C");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 0);
        }
    }

    EXPECT_TRUE(foundB && foundC);
}

TEST(HappensBeforeAnalysisTest, visit_sequence_blocks_WAR) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& input_node2 = builder.add_access(block2, "C");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_sequence(users, builder_opt.subject().root(), open_reads,
                            open_reads_after_writes, closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 2);
    EXPECT_EQ(open_reads_after_writes.size(), 2);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    bool foundA = false;
    bool foundC = false;

    for (auto entry : open_reads) {
        if (entry->container() == "A") {
            foundA = true;
            EXPECT_EQ(entry->use(), analysis::Use::READ);
            EXPECT_EQ(entry->container(), "A");
            EXPECT_EQ(entry->element(), &input_node);
        } else if (entry->container() == "C") {
            foundC = true;
            EXPECT_EQ(entry->use(), analysis::Use::READ);
            EXPECT_EQ(entry->container(), "C");
            EXPECT_EQ(entry->element(), &input_node2);
        }
    }

    EXPECT_TRUE(foundA && foundC);

    foundA = false;
    bool foundB = false;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "A") {
            foundA = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "A");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 0);
        }
    }

    EXPECT_TRUE(foundA && foundB);
}

TEST(HappensBeforeAnalysisTest, visit_sequence_blocks_WAW) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "B");
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& input_node2 = builder.add_access(block2, "C");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_sequence(users, builder_opt.subject().root(), open_reads,
                            open_reads_after_writes, closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 2);
    EXPECT_EQ(open_reads_after_writes.size(), 1);
    EXPECT_EQ(closed_reads_after_write.size(), 1);

    bool foundB = false;
    bool foundC = false;

    for (auto entry : open_reads) {
        if (entry->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry->use(), analysis::Use::READ);
            EXPECT_EQ(entry->container(), "B");
            EXPECT_EQ(entry->element(), &input_node);
        } else if (entry->container() == "C") {
            foundC = true;
            EXPECT_EQ(entry->use(), analysis::Use::READ);
            EXPECT_EQ(entry->container(), "C");
            EXPECT_EQ(entry->element(), &input_node2);
        }
    }

    EXPECT_TRUE(foundB && foundC);

    auto write = *open_reads_after_writes.begin();
    auto closed = *closed_reads_after_write.begin();

    EXPECT_EQ(write.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write.first->container(), "A");
    EXPECT_EQ(write.first->element(), &output_node2);
    EXPECT_EQ(write.second.size(), 0);

    EXPECT_EQ(closed.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(closed.first->container(), "A");
    EXPECT_EQ(closed.first->element(), &output_node);
    EXPECT_EQ(closed.second.size(), 0);
}

TEST(HappensBeforeAnalysisTest, visit_sequence_for_loop) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& for_loop = builder.add_for(
        root, symbolic::symbol("i"), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0), symbolic::add(symbolic::symbol("i"), symbolic::integer(1)));

    auto& block2 = builder.add_block(for_loop.root());
    auto& input_node2 = builder.add_access(block2, "i");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_sequence(users, builder_opt.subject().root(), open_reads,
                            open_reads_after_writes, closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 1);
    EXPECT_EQ(open_reads_after_writes.size(), 4);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    auto read = *open_reads.begin();

    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &input_node);

    bool foundB = false;
    bool foundA = false;
    int both_i = 0;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "A") {
            foundA = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "A");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "i") {
            both_i++;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "i");
            EXPECT_EQ(entry.first->element(), &for_loop);
            EXPECT_EQ(entry.second.size(), 3);
        }
    }

    EXPECT_TRUE(foundB && foundA);
    EXPECT_EQ(both_i, 2);
}

TEST(HappensBeforeAnalysisTest, visit_sequence_while_loop) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& while_loop = builder.add_while(root);

    auto& block2 = builder.add_block(while_loop.root());
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_sequence(users, builder_opt.subject().root(), open_reads,
                            open_reads_after_writes, closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 1);
    EXPECT_EQ(open_reads_after_writes.size(), 2);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    auto read = *open_reads.begin();

    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &input_node);

    bool foundB = false;
    bool foundA = false;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->container(), "B");
        } else if (entry.first->container() == "A") {
            foundA = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "A");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 0);
        }
    }

    EXPECT_TRUE(foundB && foundA);
}

TEST(HappensBeforeAnalysisTest, visit_sequence_if_else_complete) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{symbolic::symbol("C"), symbolic::symbol("B")}});
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& if_else = builder.add_if_else(root);

    auto& case1 =
        builder.add_case(if_else, symbolic::Lt(symbolic::symbol("B"), symbolic::integer(10)));

    auto& block2 = builder.add_block(case1);
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "C");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto& case2 =
        builder.add_case(if_else, symbolic::Le(symbolic::integer(10), symbolic::symbol("B")));

    auto& block3 = builder.add_block(case2);
    auto& input_node3 = builder.add_access(block3, "B");
    auto& output_node3 = builder.add_access(block3, "C");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block3, tasklet3, "_out", output_node3, "void", {symbolic::integer(0)});
    builder.add_memlet(block3, input_node3, "void", tasklet3, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_sequence(users, builder_opt.subject().root(), open_reads,
                            open_reads_after_writes, closed_reads_after_write);

    // Check result

    EXPECT_EQ(open_reads.size(), 1);
    EXPECT_EQ(open_reads_after_writes.size(), 3);
    EXPECT_EQ(closed_reads_after_write.size(), 1);

    auto read = *open_reads.begin();

    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &input_node);

    bool foundB = false;
    bool foundB_left = false;
    bool foundB_right = false;
    int countB_cond = 0;
    bool foundB_trans = false;
    bool foundC_left = false;
    bool foundC_right = false;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 4);
            for (auto entry2 : entry.second) {
                if (entry2->element() == &input_node2) {
                    foundB_left = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &input_node3) {
                    foundB_right = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &if_else) {
                    countB_cond++;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &builder_opt.subject().root().at(0).second) {
                    foundB_trans = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                }
            }
        } else if (entry.first->container() == "C") {
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            if (entry.first->element() == &output_node2) {
                EXPECT_EQ(entry.second.size(), 0);
                foundC_left = true;
            } else if (entry.first->element() == &output_node3) {
                EXPECT_EQ(entry.second.size(), 0);
                foundC_right = true;
            }
        }
    }

    EXPECT_EQ(countB_cond, 1);
    EXPECT_TRUE(foundB && foundB_left && foundB_right && foundB_trans);
    EXPECT_TRUE(foundC_left && foundC_right);

    auto closed = *closed_reads_after_write.begin();

    EXPECT_EQ(closed.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(closed.first->container(), "C");
    EXPECT_EQ(closed.first->element(), &builder_opt.subject().root().at(0).second);
    EXPECT_EQ(closed.second.size(), 0);
}

TEST(HappensBeforeAnalysisTest, visit_sequence_if_else_incomplete) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{symbolic::symbol("C"), symbolic::symbol("B")}});
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& if_else = builder.add_if_else(root);

    auto& case1 =
        builder.add_case(if_else, symbolic::Lt(symbolic::symbol("B"), symbolic::integer(10)));

    auto& block2 = builder.add_block(case1);
    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "C");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto& case2 =
        builder.add_case(if_else, symbolic::Lt(symbolic::integer(10), symbolic::symbol("B")));

    auto& block3 = builder.add_block(case2);
    auto& input_node3 = builder.add_access(block3, "B");
    auto& output_node3 = builder.add_access(block3, "C");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block3, tasklet3, "_out", output_node3, "void", {symbolic::integer(0)});
    builder.add_memlet(block3, input_node3, "void", tasklet3, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_sequence(users, builder_opt.subject().root(), open_reads,
                            open_reads_after_writes, closed_reads_after_write);

    // Check result

    EXPECT_EQ(open_reads.size(), 1);
    EXPECT_EQ(open_reads_after_writes.size(), 4);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    auto read = *open_reads.begin();

    EXPECT_EQ(read->use(), analysis::Use::READ);
    EXPECT_EQ(read->container(), "A");
    EXPECT_EQ(read->element(), &input_node);

    bool foundB = false;
    bool foundB_left = false;
    bool foundB_right = false;
    int countB_cond = 0;
    bool foundB_trans = false;
    bool foundC = false;
    bool foundC_left = false;
    bool foundC_right = false;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 4);
            for (auto entry2 : entry.second) {
                if (entry2->element() == &input_node2) {
                    foundB_left = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &input_node3) {
                    foundB_right = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &if_else) {
                    countB_cond++;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                } else if (entry2->element() == &builder_opt.subject().root().at(0).second) {
                    foundB_trans = true;
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "B");
                }
            }
        } else if (entry.first->container() == "C") {
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            if (entry.first->element() == &output_node2) {
                EXPECT_EQ(entry.second.size(), 0);
                foundC_left = true;
            } else if (entry.first->element() == &output_node3) {
                EXPECT_EQ(entry.second.size(), 0);
                foundC_right = true;
            } else if (entry.first->element() == &builder_opt.subject().root().at(0).second) {
                EXPECT_EQ(entry.second.size(), 0);
                foundC = true;
            }
        }
    }

    EXPECT_EQ(countB_cond, 1);
    EXPECT_TRUE(foundB && foundB_left && foundB_right && foundB_trans);
    EXPECT_TRUE(foundC_left && foundC_right);
}

TEST(HappensBeforeAnalysisTest, visit_sequence_transition) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root, {{symbolic::symbol("A"), symbolic::symbol("C")}});
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);

    auto& input_node2 = builder.add_access(block2, "B");
    auto& output_node2 = builder.add_access(block2, "C");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block2, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block2, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> open_reads;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        open_reads_after_writes;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>>
        closed_reads_after_write;

    analysis.visit_sequence(users, builder_opt.subject().root(), open_reads,
                            open_reads_after_writes, closed_reads_after_write);

    // Check result
    EXPECT_EQ(open_reads.size(), 2);
    EXPECT_EQ(open_reads_after_writes.size(), 3);
    EXPECT_EQ(closed_reads_after_write.size(), 0);

    bool foundA = false;
    bool foundC = false;

    for (auto entry : open_reads) {
        if (entry->container() == "A") {
            foundA = true;
            EXPECT_EQ(entry->use(), analysis::Use::READ);
            EXPECT_EQ(entry->container(), "A");
            EXPECT_EQ(entry->element(), &input_node);
        } else if (entry->container() == "C") {
            foundC = true;
            EXPECT_EQ(entry->use(), analysis::Use::READ);
            EXPECT_EQ(entry->container(), "C");
            EXPECT_EQ(entry->element(), &builder_opt.subject().root().at(0).second);
        }
    }

    EXPECT_TRUE(foundA && foundC);

    foundA = false;
    bool foundB = false;
    foundC = false;

    for (auto entry : open_reads_after_writes) {
        if (entry.first->container() == "B") {
            foundB = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.first->element(), &output_node);
            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->container(), "B");
        } else if (entry.first->container() == "C") {
            foundC = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "C");
            EXPECT_EQ(entry.first->element(), &output_node2);
            EXPECT_EQ(entry.second.size(), 0);
        } else if (entry.first->container() == "A") {
            foundA = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "A");
            EXPECT_EQ(entry.first->element(), &builder_opt.subject().root().at(0).second);
            EXPECT_EQ(entry.second.size(), 0);
        }
    }

    EXPECT_TRUE(foundA && foundB && foundC);
}

TEST(HappensBeforeAnalysisTest, visit_sdfg) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("B", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("C", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet, "_out", output_node, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node, "void", tasklet, "_in", {symbolic::integer(0)});

    auto& input_node3 = output_node;
    auto& output_node3 = builder.add_access(block, "C");
    auto& tasklet3 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet3, "_out", output_node3, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node3, "void", tasklet3, "_in", {symbolic::integer(0)});

    auto& input_node2 = output_node3;
    auto& output_node2 = builder.add_access(block, "B");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                         {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                         {{"_in", types::Scalar(types::PrimitiveType::Int32)}});
    builder.add_memlet(block, tasklet2, "_out", output_node2, "void", {symbolic::integer(0)});
    builder.add_memlet(block, input_node2, "void", tasklet2, "_in", {symbolic::integer(0)});

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    // Check result
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> reads_after_writes_A =
        analysis.reads_after_writes("A");
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> reads_after_writes_B =
        analysis.reads_after_writes("B");
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> reads_after_writes_C =
        analysis.reads_after_writes("C");

    EXPECT_EQ(reads_after_writes_A.size(), 0);
    EXPECT_EQ(reads_after_writes_B.size(), 2);
    EXPECT_EQ(reads_after_writes_C.size(), 1);

    bool foundB_first = false;
    bool foundB_second = false;
    bool foundC = false;

    for (auto& entry : reads_after_writes_B) {
        if (entry.first->element() == &output_node) {
            foundB_first = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.second.size(), 1);
            EXPECT_EQ((*entry.second.begin())->container(), "B");
            EXPECT_EQ((*entry.second.begin())->element(), &input_node3);
        } else if (entry.first->element() == &output_node2) {
            foundB_second = true;
            EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
            EXPECT_EQ(entry.first->container(), "B");
            EXPECT_EQ(entry.second.size(), 0);
        }
    }

    for (auto& entry : reads_after_writes_C) {
        foundC = true;
        EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
        EXPECT_EQ(entry.first->container(), "C");
        EXPECT_EQ(entry.second.size(), 1);
        EXPECT_EQ((*entry.second.begin())->container(), "C");
        EXPECT_EQ((*entry.second.begin())->element(), &input_node2);
    }

    EXPECT_TRUE(foundB_first && foundB_second && foundC);
}

TEST(HappensBeforeAnalysisTest, propagate_open_read_out_of_while) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    builder.add_container("_0", types::Pointer(types::Scalar(types::PrimitiveType::Double)), true);
    builder.add_container("_7", types::Pointer(types::Scalar(types::PrimitiveType::Double)));

    builder.add_container("_4", types::Scalar(types::PrimitiveType::Int32));
    auto sym = symbolic::symbol("_4");

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{sym, symbolic::integer(0)}});
    auto& outer_if_else = builder.add_if_else(root);
    auto& outer_case1 = builder.add_case(outer_if_else, symbolic::__false__());
    auto& outer_case2 = builder.add_case(outer_if_else, symbolic::__true__());

    auto& outer_loop = builder.add_while(outer_case2);
    auto& outer_body = outer_loop.root();
    auto& outer_block = builder.add_block(outer_body);

    auto& outer_input_node = builder.add_access(outer_block, "_1");
    auto& outer_output_node = builder.add_access(outer_block, "_16");
    auto& memlet =
        builder.add_memlet(outer_block, outer_input_node, "void", outer_output_node, "refs", {sym});

    auto& increment_block =
        builder.add_block(outer_body, {{sym, symbolic::add(sym, symbolic::integer(1))}});
    auto& outer_cont_break = builder.add_if_else(outer_body);
    auto& outer_cont_case =
        builder.add_case(outer_cont_break, symbolic::Lt(sym, symbolic::integer(10)));
    auto& outer_cont = builder.add_continue(outer_cont_case);
    auto& outer_break_case =
        builder.add_case(outer_cont_break, symbolic::Ge(sym, symbolic::integer(10)));
    auto& outer_break = builder.add_break(outer_break_case);

    auto sdfg = builder.move();

    // Run analysis
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto& analysis = analysis_manager.get<analysis::HappensBeforeAnalysis>();
    auto& users = analysis_manager.get<analysis::Users>();

    // Check result
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> reads_after_writes_4 =
        analysis.reads_after_writes("_4");

    EXPECT_EQ(reads_after_writes_4.size(), 2);
    for (auto& entry : reads_after_writes_4) {
        EXPECT_EQ(entry.first->use(), analysis::Use::WRITE);
        EXPECT_EQ(entry.first->container(), "_4");

        if (entry.first->element() == &root.at(0).second) {
            EXPECT_EQ(entry.second.size(), 2);
            for (auto& entry2 : entry.second) {
                if (entry2->element() == &memlet) {
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "_4");
                } else if (entry2->element() == &outer_body.at(1).second) {
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "_4");
                } else {
                    EXPECT_FALSE(true);
                }
            }
        } else if (entry.first->element() == &outer_body.at(1).second) {
            EXPECT_EQ(entry.second.size(), 3);
            for (auto& entry2 : entry.second) {
                if (entry2->element() == &memlet) {
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "_4");
                } else if (entry2->element() == &outer_cont_break) {
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "_4");
                } else if (entry2->element() == &outer_body.at(1).second) {
                    EXPECT_EQ(entry2->use(), analysis::Use::READ);
                    EXPECT_EQ(entry2->container(), "_4");
                } else {
                    EXPECT_FALSE(true);
                }
            }
        } else {
            EXPECT_FALSE(true);
        }
    }
}
