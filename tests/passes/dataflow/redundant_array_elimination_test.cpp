#include "sdfg/passes/dataflow/redundant_array_elimination.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(RedundantArrayElimination, ArrayLate) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType::CPU);

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::integer(5));
    types::Array array2(array, symbolic::integer(5));
    types::Array array3(array2, symbolic::integer(5));
    types::Array array4(array3, symbolic::integer(5));

    builder.add_container("test_in", array4, true);
    builder.add_container("test_out", array4);
    builder.add_container("a", scalar);
    builder.add_container("b", scalar);
    builder.add_container("c", scalar);
    builder.add_container("d", scalar);
    builder.add_container("e", scalar);

    auto& access_in = builder.add_access(block, "test_in");
    auto& access_out = builder.add_access(block, "test_out");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", scalar},
                                        {{"_in", scalar}});

    auto& memlet_in = builder.add_memlet(block, access_in, "void", tasklet, "_in",
                                         {symbolic::symbol("a"), symbolic::symbol("b"),
                                          symbolic::symbol("c"), symbolic::symbol("d")});

    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void",
                                          {symbolic::symbol("a"), symbolic::symbol("b"),
                                           symbolic::symbol("c"), symbolic::symbol("e")});
    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::RedundantArrayElimination pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    EXPECT_TRUE(sdfg->containers().size() == 7);
    auto& test_out = sdfg->type("test_out");
    auto atype = dynamic_cast<const types::Array*>(&test_out);
    EXPECT_TRUE(atype != nullptr);
    EXPECT_TRUE(symbolic::eq(atype->num_elements(), symbolic::integer(5)));
    auto atype2 = dynamic_cast<const types::Array*>(&atype->element_type());
    EXPECT_TRUE(atype2 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype2->num_elements(), symbolic::integer(5)));
    auto atype3 = dynamic_cast<const types::Array*>(&atype2->element_type());
    EXPECT_TRUE(atype3 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype3->num_elements(), symbolic::integer(5)));
    auto atype4 = dynamic_cast<const types::Scalar*>(&atype3->element_type());
    EXPECT_TRUE(atype4 != nullptr);

    auto& subset = memlet_out.subset();
    EXPECT_TRUE(subset.size() == 3);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("a")));
    EXPECT_TRUE(symbolic::eq(subset[1], symbolic::symbol("b")));
    EXPECT_TRUE(symbolic::eq(subset[2], symbolic::symbol("c")));
}

TEST(RedundantArrayElimination, ArrayEarly) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType::CPU);

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::integer(5));
    types::Array array2(array, symbolic::integer(5));
    types::Array array3(array2, symbolic::integer(5));
    types::Array array4(array3, symbolic::integer(5));

    builder.add_container("test_in", array4, true);
    builder.add_container("test_out", array4);
    builder.add_container("a", scalar);
    builder.add_container("b", scalar);
    builder.add_container("c", scalar);
    builder.add_container("d", scalar);
    builder.add_container("e", scalar);

    auto& access_in = builder.add_access(block, "test_in");
    auto& access_out = builder.add_access(block, "test_out");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", scalar},
                                        {{"_in", scalar}});

    auto& memlet_in = builder.add_memlet(block, access_in, "void", tasklet, "_in",
                                         {symbolic::symbol("a"), symbolic::symbol("b"),
                                          symbolic::symbol("c"), symbolic::symbol("d")});

    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void",
                                          {symbolic::symbol("e"), symbolic::symbol("a"),
                                           symbolic::symbol("b"), symbolic::symbol("c")});
    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::RedundantArrayElimination pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    EXPECT_TRUE(sdfg->containers().size() == 7);
    auto& test_out = sdfg->type("test_out");
    auto atype = dynamic_cast<const types::Array*>(&test_out);
    EXPECT_TRUE(atype != nullptr);
    EXPECT_TRUE(symbolic::eq(atype->num_elements(), symbolic::integer(5)));
    auto atype2 = dynamic_cast<const types::Array*>(&atype->element_type());
    EXPECT_TRUE(atype2 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype2->num_elements(), symbolic::integer(5)));
    auto atype3 = dynamic_cast<const types::Array*>(&atype2->element_type());
    EXPECT_TRUE(atype3 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype3->num_elements(), symbolic::integer(5)));
    auto atype4 = dynamic_cast<const types::Scalar*>(&atype3->element_type());
    EXPECT_TRUE(atype4 != nullptr);

    auto& subset = memlet_out.subset();
    EXPECT_TRUE(subset.size() == 3);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("a")));
    EXPECT_TRUE(symbolic::eq(subset[1], symbolic::symbol("b")));
    EXPECT_TRUE(symbolic::eq(subset[2], symbolic::symbol("c")));
}

TEST(RedundantArrayElimination, ArrayMiddle) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType::CPU);

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::integer(5));
    types::Array array2(array, symbolic::integer(5));
    types::Array array3(array2, symbolic::integer(5));
    types::Array array4(array3, symbolic::integer(5));

    builder.add_container("test_in", array4, true);
    builder.add_container("test_out", array4);
    builder.add_container("a", scalar);
    builder.add_container("b", scalar);
    builder.add_container("c", scalar);
    builder.add_container("d", scalar);
    builder.add_container("e", scalar);

    auto& access_in = builder.add_access(block, "test_in");
    auto& access_out = builder.add_access(block, "test_out");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", scalar},
                                        {{"_in", scalar}});

    auto& memlet_in = builder.add_memlet(block, access_in, "void", tasklet, "_in",
                                         {symbolic::symbol("a"), symbolic::symbol("b"),
                                          symbolic::symbol("c"), symbolic::symbol("d")});

    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void",
                                          {symbolic::symbol("a"), symbolic::symbol("e"),
                                           symbolic::symbol("b"), symbolic::symbol("c")});
    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::RedundantArrayElimination pass;
    EXPECT_TRUE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    EXPECT_TRUE(sdfg->containers().size() == 7);
    auto& test_out = sdfg->type("test_out");
    auto atype = dynamic_cast<const types::Array*>(&test_out);
    EXPECT_TRUE(atype != nullptr);
    EXPECT_TRUE(symbolic::eq(atype->num_elements(), symbolic::integer(5)));
    auto atype2 = dynamic_cast<const types::Array*>(&atype->element_type());
    EXPECT_TRUE(atype2 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype2->num_elements(), symbolic::integer(5)));
    auto atype3 = dynamic_cast<const types::Array*>(&atype2->element_type());
    EXPECT_TRUE(atype3 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype3->num_elements(), symbolic::integer(5)));
    auto atype4 = dynamic_cast<const types::Scalar*>(&atype3->element_type());
    EXPECT_TRUE(atype4 != nullptr);

    auto& subset = memlet_out.subset();
    EXPECT_TRUE(subset.size() == 3);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("a")));
    EXPECT_TRUE(symbolic::eq(subset[1], symbolic::symbol("b")));
    EXPECT_TRUE(symbolic::eq(subset[2], symbolic::symbol("c")));
}

TEST(RedundantArrayElimination, Negative) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType::CPU);

    auto& block = builder.add_block(builder.subject().root());

    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Array array(scalar, symbolic::integer(5));
    types::Array array2(array, symbolic::integer(5));
    types::Array array3(array2, symbolic::integer(5));
    types::Array array4(array3, symbolic::integer(5));

    builder.add_container("test_in", array4, true);
    builder.add_container("test_out", array4);
    builder.add_container("a", scalar);
    builder.add_container("b", scalar);
    builder.add_container("c", scalar);
    builder.add_container("d", scalar);

    auto& access_in = builder.add_access(block, "test_in");
    auto& access_out = builder.add_access(block, "test_out");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out", scalar},
                                        {{"_in", scalar}});

    auto& memlet_in = builder.add_memlet(block, access_in, "void", tasklet, "_in",
                                         {symbolic::symbol("a"), symbolic::symbol("b"),
                                          symbolic::symbol("c"), symbolic::symbol("d")});

    auto& memlet_out = builder.add_memlet(block, tasklet, "_out", access_out, "void",
                                          {symbolic::symbol("a"), symbolic::symbol("b"),
                                           symbolic::symbol("c"), symbolic::symbol("d")});
    auto sdfg = builder.move();

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::RedundantArrayElimination pass;
    EXPECT_FALSE(pass.run_pass(builder_opt, analysis_manager));
    sdfg = builder_opt.move();

    EXPECT_TRUE(sdfg->containers().size() == 6);
    auto& test_out = sdfg->type("test_out");
    auto atype = dynamic_cast<const types::Array*>(&test_out);
    EXPECT_TRUE(atype != nullptr);
    EXPECT_TRUE(symbolic::eq(atype->num_elements(), symbolic::integer(5)));
    auto atype2 = dynamic_cast<const types::Array*>(&atype->element_type());
    EXPECT_TRUE(atype2 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype2->num_elements(), symbolic::integer(5)));
    auto atype3 = dynamic_cast<const types::Array*>(&atype2->element_type());
    EXPECT_TRUE(atype3 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype3->num_elements(), symbolic::integer(5)));
    auto atype4 = dynamic_cast<const types::Array*>(&atype3->element_type());
    EXPECT_TRUE(atype4 != nullptr);
    EXPECT_TRUE(symbolic::eq(atype3->num_elements(), symbolic::integer(5)));
    auto atype5 = dynamic_cast<const types::Scalar*>(&atype4->element_type());
    EXPECT_TRUE(atype5 != nullptr);

    auto& subset = memlet_out.subset();
    EXPECT_TRUE(subset.size() == 4);
    EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("a")));
    EXPECT_TRUE(symbolic::eq(subset[1], symbolic::symbol("b")));
    EXPECT_TRUE(symbolic::eq(subset[2], symbolic::symbol("c")));
    EXPECT_TRUE(symbolic::eq(subset[3], symbolic::symbol("d")));
}
