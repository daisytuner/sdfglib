#include <gtest/gtest.h>
#include <sdfg/optimizations/optimizer.h>

#include <fstream>
#include <memory>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_tiling.h"

using namespace sdfg;

class OptimizerTest : public ::testing::Test {
   protected:
    std::unique_ptr<builder::StructuredSDFGBuilder> builder_;
    std::unique_ptr<analysis::AnalysisManager> analysis_manager_;
    structured_control_flow::For* loop_;

    void SetUp() override {
        builder_ = std::make_unique<builder::StructuredSDFGBuilder>("sdfg_test", FunctionType_CPU);
        auto& sdfg = builder_->subject();
        auto& root = sdfg.root();

        // Add containers
        types::Scalar base_desc(types::PrimitiveType::Float);
        types::Pointer desc(base_desc);
        builder_->add_container("A", desc, true);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder_->add_container("N", sym_desc, true);
        builder_->add_container("i", sym_desc);

        // Define loop
        auto bound = symbolic::symbol("N");
        auto indvar = symbolic::symbol("i");
        auto init = symbolic::integer(0);
        auto condition = symbolic::Lt(indvar, bound);
        auto update = symbolic::add(indvar, symbolic::integer(1));

        loop_ = &builder_->add_for(root, indvar, condition, init, update);
        auto& body = loop_->root();

        // Add computation
        auto& block = builder_->add_block(body);
        auto& A_in = builder_->add_access(block, "A");
        auto& A_out = builder_->add_access(block, "A");
        auto& tasklet = builder_->add_tasklet(block, data_flow::TaskletCode::assign,
                                              {"_out", base_desc}, {{"_in", base_desc}});
        builder_->add_memlet(block, A_in, "void", tasklet, "_in", {symbolic::symbol("i")});
        builder_->add_memlet(block, tasklet, "_out", A_out, "void", {symbolic::symbol("i")});

        analysis_manager_ = std::make_unique<analysis::AnalysisManager>(builder_->subject());
    }
    void TearDown() override {
        // Cleanup if necessary
    };
};

TEST_F(OptimizerTest, Apply_LoopTiling) {
    optimizations::Optimizer optimizer;

    EXPECT_TRUE(loop_ != nullptr);
    EXPECT_TRUE(analysis_manager_ != nullptr);
    EXPECT_NO_THROW(
        optimizer.apply<transformations::LoopTiling>(*builder_, *analysis_manager_, *loop_, 32));
}

TEST_F(OptimizerTest, Apply_InvalidTransformation) {
    optimizations::Optimizer optimizer;

    EXPECT_THROW(
        optimizer.apply<transformations::LoopTiling>(*builder_, *analysis_manager_, *loop_, 0),
        transformations::InvalidTransformationException);
}

TEST_F(OptimizerTest, Save_SingleTransformation) {
    optimizations::Optimizer optimizer;

    size_t loop_id = loop_->element_id();

    EXPECT_NO_THROW(
        optimizer.apply<transformations::LoopTiling>(*builder_, *analysis_manager_, *loop_, 32));

    // Use temporary file to save the transformation
    std::filesystem::path tmp_file =
        std::filesystem::temp_directory_path() / "Save_SingleTransformation.json";
    EXPECT_NO_THROW(optimizer.save(tmp_file));
    EXPECT_TRUE(std::filesystem::exists(tmp_file));

    std::ifstream file(tmp_file);
    nlohmann::json j = nlohmann::json::array();
    file >> j;
    file.close();

    std::cout << j.dump(4) << std::endl;

    EXPECT_TRUE(j.is_array());
    EXPECT_EQ(j.size(), 1);
    EXPECT_EQ(j[0]["transformation_type"], "LoopTiling");
    EXPECT_EQ(j[0]["tile_size"], 32);
    EXPECT_EQ(j[0]["loop_element_id"], loop_id);
}

TEST_F(OptimizerTest, Replay_Transformations) {
    builder::StructuredSDFGBuilder replay_builder("sdfg_test", FunctionType_CPU);

    optimizations::Optimizer optimizer;

    nlohmann::json j = nlohmann::json::array();
    j.push_back({{"loop_element_id", 1}, {"tile_size", 32}, {"transformation_type", "LoopTiling"}});

    EXPECT_NO_THROW(optimizer.replay(*builder_, *analysis_manager_, j));
}
