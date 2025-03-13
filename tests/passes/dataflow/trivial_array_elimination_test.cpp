#include "sdfg/passes/dataflow/trivial_array_elimination.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(TrivialArrayElimination, ReadArrayLate) {
    builder::StructuredSDFGBuilder builder("sdfg");

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Scalar scalar2(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::one());
    types::Array array2(array, symbolic::integer(2));
    types::Array array3(array2, symbolic::integer(3));
    types::Array array4(array3, symbolic::integer(4));

    builder.add_container("test_in", array4);
    builder.add_container("test_out", scalar2);

    auto& access_in = builder.add_access(block, "test_in");
    auto& access_out = builder.add_access(block, "test_out");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", scalar2},
                                        {{"_in", scalar}});

    auto& memlet_in = builder.add_memlet(
        block, access_in, "void", tasklet, "_in",
        {symbolic::integer(3), symbolic::integer(2), symbolic::integer(1), symbolic::integer(0)});

    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void", {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::TrivialArrayElimination pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    EXPECT_TRUE(sdfg->containers().size() == 2);
    auto& test_in = sdfg->type("test_in");
    auto atype = dynamic_cast<const types::Array*>(&test_in);
    EXPECT_TRUE(atype != nullptr);
    EXPECT_TRUE(symbolic::eq(atype->num_elements(), symbolic::integer(4)));
    auto atype2 = dynamic_cast<const types::Array*>(&atype->element_type());
    EXPECT_TRUE(atype2 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype2->num_elements(), symbolic::integer(3)));
    auto atype3 = dynamic_cast<const types::Array*>(&atype2->element_type());
    EXPECT_TRUE(atype3 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype3->num_elements(), symbolic::integer(2)));
    auto atype4 = dynamic_cast<const types::Scalar*>(&atype3->element_type());
    EXPECT_TRUE(atype4 != nullptr);

    auto& subset = memlet_in.subset();
    EXPECT_TRUE(subset.size() == 3);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(3)));
    EXPECT_TRUE(symbolic::eq(subset[1], symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(subset[2], symbolic::integer(1)));
}

TEST(TrivialArrayElimination, ReadArrayEarly) {
    builder::StructuredSDFGBuilder builder("sdfg");

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Scalar scalar2(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::integer(4));
    types::Array array2(array, symbolic::integer(3));
    types::Array array3(array2, symbolic::integer(2));
    types::Array array4(array3, symbolic::integer(1));

    builder.add_container("test_in", array4);
    builder.add_container("test_out", scalar2);

    auto& access_in = builder.add_access(block, "test_in");
    auto& access_out = builder.add_access(block, "test_out");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", scalar2},
                                        {{"_in", scalar}});

    auto& memlet_in = builder.add_memlet(
        block, access_in, "void", tasklet, "_in",
        {symbolic::integer(0), symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)});

    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void", {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::TrivialArrayElimination pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    EXPECT_TRUE(sdfg->containers().size() == 2);
    auto& test_in = sdfg->type("test_in");
    auto atype = dynamic_cast<const types::Array*>(&test_in);
    EXPECT_TRUE(atype != nullptr);
    EXPECT_TRUE(symbolic::eq(atype->num_elements(), symbolic::integer(2)));
    auto atype2 = dynamic_cast<const types::Array*>(&atype->element_type());
    EXPECT_TRUE(atype2 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype2->num_elements(), symbolic::integer(3)));
    auto atype3 = dynamic_cast<const types::Array*>(&atype2->element_type());
    EXPECT_TRUE(atype3 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype3->num_elements(), symbolic::integer(4)));
    auto atype4 = dynamic_cast<const types::Scalar*>(&atype3->element_type());
    EXPECT_TRUE(atype4 != nullptr);

    auto& subset = memlet_in.subset();
    EXPECT_TRUE(subset.size() == 3);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(subset[1], symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(subset[2], symbolic::integer(3)));
}

TEST(TrivialArrayElimination, ReadArrayMiddle) {
    builder::StructuredSDFGBuilder builder("sdfg");

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Scalar scalar2(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::integer(4));
    types::Array array2(array, symbolic::integer(3));
    types::Array array3(array2, symbolic::integer(1));
    types::Array array4(array3, symbolic::integer(2));

    builder.add_container("test_in", array4);
    builder.add_container("test_out", scalar2);

    auto& access_in = builder.add_access(block, "test_in");
    auto& access_out = builder.add_access(block, "test_out");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", scalar2},
                                        {{"_in", scalar}});

    auto& memlet_in = builder.add_memlet(
        block, access_in, "void", tasklet, "_in",
        {symbolic::integer(1), symbolic::integer(0), symbolic::integer(2), symbolic::integer(3)});

    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void", {});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::TrivialArrayElimination pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    EXPECT_TRUE(sdfg->containers().size() == 2);
    auto& test_in = sdfg->type("test_in");
    auto atype = dynamic_cast<const types::Array*>(&test_in);
    EXPECT_TRUE(atype != nullptr);
    EXPECT_TRUE(symbolic::eq(atype->num_elements(), symbolic::integer(2)));
    auto atype2 = dynamic_cast<const types::Array*>(&atype->element_type());
    EXPECT_TRUE(atype2 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype2->num_elements(), symbolic::integer(3)));
    auto atype3 = dynamic_cast<const types::Array*>(&atype2->element_type());
    EXPECT_TRUE(atype3 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype3->num_elements(), symbolic::integer(4)));
    auto atype4 = dynamic_cast<const types::Scalar*>(&atype3->element_type());
    EXPECT_TRUE(atype4 != nullptr);

    auto& subset = memlet_in.subset();
    EXPECT_TRUE(subset.size() == 3);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(subset[1], symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(subset[2], symbolic::integer(3)));
}

TEST(TrivialArrayElimination, WriteArrayLate) {
    builder::StructuredSDFGBuilder builder("sdfg");

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Scalar scalar2(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::one());
    types::Array array2(array, symbolic::integer(2));
    types::Array array3(array2, symbolic::integer(3));
    types::Array array4(array3, symbolic::integer(4));

    builder.add_container("test_in", scalar2);
    builder.add_container("test_out", array4);

    auto& access_in = builder.add_access(block, "test_in");
    auto& access_out = builder.add_access(block, "test_out");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", scalar2},
                                        {{"_in", scalar}});

    auto& memlet_in = builder.add_memlet(block, access_in, "void", tasklet, "_in", {});

    auto& memlet_out = builder.add_memlet(
        block, tasklet, "_out", access_out, "void",
        {symbolic::integer(3), symbolic::integer(2), symbolic::integer(1), symbolic::integer(0)});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::TrivialArrayElimination pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    EXPECT_TRUE(sdfg->containers().size() == 2);
    auto& test_out = sdfg->type("test_out");
    auto atype = dynamic_cast<const types::Array*>(&test_out);
    EXPECT_TRUE(atype != nullptr);
    EXPECT_TRUE(symbolic::eq(atype->num_elements(), symbolic::integer(4)));
    auto atype2 = dynamic_cast<const types::Array*>(&atype->element_type());
    EXPECT_TRUE(atype2 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype2->num_elements(), symbolic::integer(3)));
    auto atype3 = dynamic_cast<const types::Array*>(&atype2->element_type());
    EXPECT_TRUE(atype3 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype3->num_elements(), symbolic::integer(2)));
    auto atype4 = dynamic_cast<const types::Scalar*>(&atype3->element_type());
    EXPECT_TRUE(atype4 != nullptr);

    auto& subset = memlet_out.subset();
    EXPECT_TRUE(subset.size() == 3);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(3)));
    EXPECT_TRUE(symbolic::eq(subset[1], symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(subset[2], symbolic::integer(1)));
}

TEST(TrivialArrayElimination, WriteArrayEarly) {
    builder::StructuredSDFGBuilder builder("sdfg");

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Scalar scalar2(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::integer(4));
    types::Array array2(array, symbolic::integer(3));
    types::Array array3(array2, symbolic::integer(2));
    types::Array array4(array3, symbolic::integer(1));

    builder.add_container("test_in", scalar2);
    builder.add_container("test_out", array4);

    auto& access_in = builder.add_access(block, "test_in");
    auto& access_out = builder.add_access(block, "test_out");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", scalar2},
                                        {{"_in", scalar}});

    auto& memlet_in = builder.add_memlet(block, access_in, "void", tasklet, "_in", {});

    auto& memlet_out = builder.add_memlet(
        block, tasklet, "_out", access_out, "void",
        {symbolic::integer(0), symbolic::integer(1), symbolic::integer(2), symbolic::integer(3)});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::TrivialArrayElimination pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    EXPECT_TRUE(sdfg->containers().size() == 2);
    auto& test_out = sdfg->type("test_out");
    auto atype = dynamic_cast<const types::Array*>(&test_out);
    EXPECT_TRUE(atype != nullptr);
    EXPECT_TRUE(symbolic::eq(atype->num_elements(), symbolic::integer(2)));
    auto atype2 = dynamic_cast<const types::Array*>(&atype->element_type());
    EXPECT_TRUE(atype2 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype2->num_elements(), symbolic::integer(3)));
    auto atype3 = dynamic_cast<const types::Array*>(&atype2->element_type());
    EXPECT_TRUE(atype3 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype3->num_elements(), symbolic::integer(4)));
    auto atype4 = dynamic_cast<const types::Scalar*>(&atype3->element_type());
    EXPECT_TRUE(atype4 != nullptr);

    auto& subset = memlet_out.subset();
    EXPECT_TRUE(subset.size() == 3);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(subset[1], symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(subset[2], symbolic::integer(3)));
}

TEST(TrivialArrayElimination, WriteArrayMiddle) {
    builder::StructuredSDFGBuilder builder("sdfg");

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Scalar scalar2(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::integer(4));
    types::Array array2(array, symbolic::integer(3));
    types::Array array3(array2, symbolic::integer(1));
    types::Array array4(array3, symbolic::integer(2));

    builder.add_container("test_in", scalar2);
    builder.add_container("test_out", array4);

    auto& access_in = builder.add_access(block, "test_in");
    auto& access_out = builder.add_access(block, "test_out");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", scalar2},
                                        {{"_in", scalar}});

    auto& memlet_in = builder.add_memlet(block, access_in, "void", tasklet, "_in", {});

    auto& memlet_out = builder.add_memlet(
        block, tasklet, "_out", access_out, "void",
        {symbolic::integer(1), symbolic::integer(0), symbolic::integer(2), symbolic::integer(3)});

    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::TrivialArrayElimination pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    EXPECT_TRUE(sdfg->containers().size() == 2);
    auto& test_out = sdfg->type("test_out");
    auto atype = dynamic_cast<const types::Array*>(&test_out);
    EXPECT_TRUE(atype != nullptr);
    EXPECT_TRUE(symbolic::eq(atype->num_elements(), symbolic::integer(2)));
    auto atype2 = dynamic_cast<const types::Array*>(&atype->element_type());
    EXPECT_TRUE(atype2 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype2->num_elements(), symbolic::integer(3)));
    auto atype3 = dynamic_cast<const types::Array*>(&atype2->element_type());
    EXPECT_TRUE(atype3 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype3->num_elements(), symbolic::integer(4)));
    auto atype4 = dynamic_cast<const types::Scalar*>(&atype3->element_type());
    EXPECT_TRUE(atype4 != nullptr);

    auto& subset = memlet_out.subset();
    EXPECT_TRUE(subset.size() == 3);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(subset[1], symbolic::integer(2)));
    EXPECT_TRUE(symbolic::eq(subset[2], symbolic::integer(3)));
}
