
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"

using namespace sdfg;

class ReplayerTest : public ::testing::Test {
protected:
    std::unique_ptr<builder::StructuredSDFGBuilder> builder_;
    nlohmann::json desc_;

    void SetUp() override {
        builder_ = std::make_unique<builder::StructuredSDFGBuilder>("sdfg_test", FunctionType_CPU);

        auto& root = builder_->subject().root();
        types::Scalar base_desc(types::PrimitiveType::Float);
        types::Array desc_1(base_desc, symbolic::integer(64));
        types::Pointer desc_2(desc_1);

        builder_->add_container("A", desc_2, true);
        builder_->add_container("B", desc_2, true);
        builder_->add_container("C", desc_2, true);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder_->add_container("K", sym_desc, true);
        builder_->add_container("N", sym_desc, true);
        builder_->add_container("M", sym_desc, true);
        builder_->add_container("i", sym_desc);
        builder_->add_container("j", sym_desc);
        builder_->add_container("k", sym_desc);

        // Define loop 1
        auto bound = symbolic::integer(64);
        auto indvar = symbolic::symbol("i");

        auto& loop = builder_->add_map(
            root,
            indvar,
            symbolic::Lt(symbolic::symbol("i"), bound),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& body = loop.root();

        // Define loop 2
        auto bound_2 = symbolic::integer(64);
        auto indvar_2 = symbolic::symbol("j");

        auto& loop_2 = builder_->add_for(
            body,
            indvar_2,
            symbolic::Lt(symbolic::symbol("j"), bound_2),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
        );

        auto& body_2 = loop_2.root();

        // Define loop 3
        auto bound_3 = symbolic::integer(64);
        auto indvar_3 = symbolic::symbol("k");

        auto& loop_3 = builder_->add_map(
            body_2,
            indvar_3,
            symbolic::Lt(symbolic::symbol("k"), bound_3),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );

        auto& body_3 = loop_3.root();

        // Add computation
        auto& block = builder_->add_block(body_3);
        auto& a_in = builder_->add_access(block, "A");
        auto& b_in = builder_->add_access(block, "B");
        auto& c_in = builder_->add_access(block, "C");
        auto& c_out = builder_->add_access(block, "C");

        {
            auto& tasklet =
                builder_->add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
            builder_
                ->add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
            builder_
                ->add_computational_memlet(block, b_in, tasklet, "_in2", {symbolic::symbol("j"), symbolic::symbol("k")});
            builder_
                ->add_computational_memlet(block, c_in, tasklet, "_in3", {symbolic::symbol("i"), symbolic::symbol("k")});
            builder_
                ->add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("i"), symbolic::symbol("k")});
        }

        // Apply transformations
        {
            auto new_sdfg = builder_->subject().clone();
            sdfg::builder::StructuredSDFGBuilder builder_clone(new_sdfg);

            sdfg::transformations::Recorder recorder;

            analysis::AnalysisManager analysis_manager(builder_clone.subject());

            auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
            auto loops = loop_analysis.loops();

            // Step 1: Tile all loops
            for (auto& loop : loops) {
                recorder.apply<sdfg::transformations::LoopTiling>(
                    builder_clone,
                    analysis_manager,
                    false,
                    *static_cast<sdfg::structured_control_flow::StructuredLoop*>(loop),
                    32
                );
            }

            /*
    for (size_t i_tile = 0; i_tile < N; i_tile += 512) {
        for (size_t i = i_tile; i < std::min(static_cast<size_t>(N), i_tile + 32); ++i) {
            for (size_t j_tile = 0; j_tile < M; j_tile += 32) {
                for (size_t j = j_tile; j < std::min(static_cast<size_t>(M), j_tile + 32); ++j) {
                    for (size_t k_tile = 0; k_tile < K; k_tile += 32) {
                        for (size_t k = k_tile;
                             k < std::min(static_cast<size_t>(K), k_tile + 32); ++k) {
                            C[i][k] += A[i][j] * B[j][k];
                        }
                    }
                }
            }
        }
    }
*/


            // Step 2: Interchange i <-> j_tile0
            auto& loop_analysis_2 = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
            auto loop_i = static_cast<sdfg::structured_control_flow::Map*>(loop_analysis_2.find_loop_by_indvar("i"));
            size_t loop_i_element_id = loop_i->element_id();
            auto loop_j_tile =
                static_cast<sdfg::structured_control_flow::For*>(loop_analysis_2.find_loop_by_indvar("j_tile0"));
            size_t loop_j_tile_element_id = loop_j_tile->element_id();
            recorder.apply<
                sdfg::transformations::LoopInterchange>(builder_clone, analysis_manager, false, *loop_i, *loop_j_tile);

            /*
    for (size_t i_tile = 0; i_tile < N; i_tile += 512) {
        for (size_t j_tile = 0; j_tile < M; j_tile += 32) {
            for (size_t i = i_tile; i < std::min(static_cast<size_t>(N), i_tile + 32); ++i) {
                for (size_t j = j_tile; j < std::min(static_cast<size_t>(M), j_tile + 32); ++j) {
                    for (size_t k_tile = 0; k_tile < K; k_tile += 32) {
                        for (size_t k = k_tile;
                             k < std::min(static_cast<size_t>(K), k_tile + 32); ++k) {
                            C[i][k] += A[i][j] * B[j][k];
                        }
                    }
                }
            }
        }
    }
*/

            // Step 3: Interchange j <-> k_tile0
            auto& loop_analysis_3 = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
            auto loop_j = static_cast<sdfg::structured_control_flow::For*>(loop_analysis_3.find_loop_by_indvar("j"));
            size_t loop_j_element_id = loop_j->element_id();
            auto loop_k_tile =
                static_cast<sdfg::structured_control_flow::Map*>(loop_analysis_3.find_loop_by_indvar("k_tile0"));
            size_t loop_k_tile_element_id = loop_k_tile->element_id();
            recorder.apply<
                sdfg::transformations::LoopInterchange>(builder_clone, analysis_manager, false, *loop_j, *loop_k_tile);

            /*
    for (size_t i_tile = 0; i_tile < N; i_tile += 512) {
        for (size_t j_tile = 0; j_tile < M; j_tile += 32) {
            for (size_t i = i_tile; i < std::min(static_cast<size_t>(N), i_tile + 32); ++i) {
                for (size_t k_tile = 0; k_tile < K; k_tile += 32) {
                    for (size_t j = j_tile; j < std::min(static_cast<size_t>(M), j_tile + 32); ++j) {
                        for (size_t k = k_tile; k < std::min(static_cast<size_t>(K), k_tile + 32); ++k) {
                            C[i][k] += A[i][j] * B[j][k];
                        }
                    }
                }
            }
        }
    }
            */

            // Step 4: Interchange i <-> k_tile0
            auto& loop_analysis_4 = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
            auto loop_i_2 = static_cast<sdfg::structured_control_flow::Map*>(loop_analysis_4.find_loop_by_indvar("i"));
            size_t loop_i_2_element_id = loop_i_2->element_id();
            auto loop_k_tile_2 =
                static_cast<sdfg::structured_control_flow::Map*>(loop_analysis_4.find_loop_by_indvar("k_tile0"));
            size_t loop_k_tile_2_element_id = loop_k_tile_2->element_id();
            recorder.apply<
                sdfg::transformations::LoopInterchange>(builder_clone, analysis_manager, false, *loop_i_2, *loop_k_tile_2);


            /*
    for (size_t i_tile = 0; i_tile < N; i_tile += 512) {
        for (size_t j_tile = 0; j_tile < M; j_tile += 32) {
            for (size_t k_tile = 0; k_tile < K; k_tile += 32) {
                for (size_t i = i_tile; i < std::min(static_cast<size_t>(N), i_tile + 32); ++i) {
                    for (size_t j = j_tile; j < std::min(static_cast<size_t>(M), j_tile + 32); ++j) {
                        for (size_t k = k_tile; k < std::min(static_cast<size_t>(K), k_tile + 32); ++k) {
                            C[i][k] += A[i][j] * B[j][k];
                        }
                    }
                }
            }
        }
    }
            */

            // Step 5: Save the transformations
            this->desc_ = recorder.history();
        }
    };

    void TearDown() override {
        // Cleanup if necessary
    };
};

TEST_F(ReplayerTest, Matmul_FMA) {
    sdfg::analysis::AnalysisManager analysis_manager(builder_->subject());
    sdfg::transformations::Replayer replayer;
    replayer.replay(*this->builder_, analysis_manager, this->desc_);

    auto& test_loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto loop_nest_tree = test_loop_analysis.loop_tree();
    EXPECT_TRUE(test_loop_analysis.find_loop_by_indvar("j") == loop_nest_tree[test_loop_analysis.find_loop_by_indvar("k")]);
    EXPECT_TRUE(test_loop_analysis.find_loop_by_indvar("i") == loop_nest_tree[test_loop_analysis.find_loop_by_indvar("j")]);
    EXPECT_TRUE(
        test_loop_analysis.find_loop_by_indvar("k_tile0") == loop_nest_tree[test_loop_analysis.find_loop_by_indvar("i")]
    );
    EXPECT_TRUE(
        test_loop_analysis.find_loop_by_indvar("j_tile0") ==
        loop_nest_tree[test_loop_analysis.find_loop_by_indvar("k_tile0")]
    );
    EXPECT_TRUE(
        test_loop_analysis.find_loop_by_indvar("i_tile0") ==
        loop_nest_tree[test_loop_analysis.find_loop_by_indvar("j_tile0")]
    );
};
