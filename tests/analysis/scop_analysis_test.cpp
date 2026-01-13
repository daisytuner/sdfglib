#include <gtest/gtest.h>

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/analysis/scop_analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/symbolic/symbolic.h>

#include <isl/ctx.h>
#include <isl/point.h>
#include <isl/set.h>
#include <isl/val.h>

using namespace sdfg;

TEST(ScopAnalysisTest, ScopBuilderTest_Tasklet) {
    builder::StructuredSDFGBuilder builder("simple_loop_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("A", int_type);
    builder.add_container("B", int_type);

    // Add tasklet
    auto& block = builder.add_block(root);
    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});
    builder.add_computational_memlet(block, in_node, tasklet, "_int", {});
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {});

    // Run Analysis
    analysis::AnalysisManager am(sdfg);

    analysis::ScopBuilder scop_builder(sdfg, root);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    auto statements = scop->statements();
    ASSERT_EQ(statements.size(), 1);

    auto* stmt = statements[0];
    isl_set* domain = stmt->domain();
    ASSERT_NE(domain, nullptr);

    // Verify properties of domain
    // It should be 0-dimensional
    EXPECT_EQ(isl_set_dim(domain, isl_dim_set), 0);

    // Verify memory accesses
    auto reads = stmt->reads();
    EXPECT_EQ(reads.size(), 1);
    auto read_access = *reads.begin();
    EXPECT_EQ(read_access->data(), "A");
    EXPECT_EQ(read_access->access_type(), analysis::AccessType::READ);
    const char* read_relation_cstr = isl_map_to_str(read_access->relation());
    std::string read_relation = read_relation_cstr;
    EXPECT_EQ(read_relation, "{ S_5[] -> [0] }");
    free((void*) read_relation_cstr);

    auto writes = stmt->writes();
    EXPECT_EQ(writes.size(), 1);
    auto write_access = *writes.begin();
    EXPECT_EQ(write_access->data(), "B");
    EXPECT_EQ(write_access->access_type(), analysis::AccessType::WRITE);
    const char* write_relation_cstr = isl_map_to_str(write_access->relation());
    std::string write_relation = write_relation_cstr;
    EXPECT_EQ(write_relation, "{ S_5[] -> [0] }");
    free((void*) write_relation_cstr);
}

TEST(ScopAnalysisTest, ScopBuilderTest_Expression) {
    builder::StructuredSDFGBuilder builder("simple_loop_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("A", int_type);
    builder.add_container("B", int_type);

    // Add tasklet
    auto& block = builder.add_block(root, {{symbolic::symbol("A"), symbolic::symbol("B")}});

    // Run Analysis
    analysis::AnalysisManager am(sdfg);

    analysis::ScopBuilder scop_builder(sdfg, root);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    auto statements = scop->statements();
    ASSERT_EQ(statements.size(), 1);

    auto* stmt = statements[0];
    isl_set* domain = stmt->domain();
    ASSERT_NE(domain, nullptr);

    // Verify properties of domain
    // It should be 0-dimensional
    EXPECT_EQ(isl_set_dim(domain, isl_dim_set), 0);
}

TEST(ScopAnalysisTest, ScopBuilderTest_LoopWithConstantBounds_StaticStatements) {
    builder::StructuredSDFGBuilder builder("simple_loop_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("A", int_type);
    builder.add_container("B", int_type);

    // Create loop: for i = 0 to 10
    auto i_sym = symbolic::symbol("i");
    auto n_val = symbolic::integer(10);
    auto init = symbolic::integer(0);
    // update: i + 1
    auto update = symbolic::add(i_sym, symbolic::integer(1));
    // condition: i < 10
    auto cond = symbolic::Lt(i_sym, n_val);

    auto& loop = builder.add_for(root, i_sym, cond, init, update);

    // Add body
    auto& block = builder.add_block(loop.root());

    // Add dummy tasklet
    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});
    builder.add_computational_memlet(block, in_node, tasklet, "_int", {});
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {});

    // Run Analysis
    analysis::AnalysisManager am(sdfg);

    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    auto statements = scop->statements();
    ASSERT_EQ(statements.size(), 1);

    auto* stmt = statements[0];
    isl_set* domain = stmt->domain();
    ASSERT_NE(domain, nullptr);

    // Verify properties of domain
    // It should be 1-dimensional
    EXPECT_EQ(isl_set_dim(domain, isl_dim_set), 1);

    // Verify name of dimension
    const char* dim_name = isl_set_get_dim_name(domain, isl_dim_set, 0);
    EXPECT_STREQ(dim_name, "i");

    // Verify bounds: i >= 0 and i < 10
    isl_ctx* ctx = scop->ctx();

    // Helper lambda for point checking
    auto check_point = [&](int val, bool expected) {
        isl_point* p = isl_point_zero(isl_set_get_space(domain));
        p = isl_point_set_coordinate_val(p, isl_dim_set, 0, isl_val_int_from_si(ctx, val));

        // Use isl_set_is_subset(from_point(p), domain)
        isl_set* p_set = isl_set_from_point(p);
        bool is_subset = isl_set_is_subset(p_set, domain);
        EXPECT_EQ(is_subset, expected) << "Value " << val << " expected " << expected;
        isl_set_free(p_set);
    };

    check_point(0, true);
    check_point(9, true);
    check_point(10, false);
    check_point(-1, false);

    // Verify memory accesses
    auto reads = stmt->reads();
    EXPECT_EQ(reads.size(), 1);
    auto read_access = *reads.begin();
    EXPECT_EQ(read_access->data(), "A");
    EXPECT_EQ(read_access->access_type(), analysis::AccessType::READ);
    const char* read_relation_cstr = isl_map_to_str(read_access->relation());
    std::string read_relation = read_relation_cstr;
    EXPECT_EQ(read_relation, "{ S_8[i] -> [0] }");
    free((void*) read_relation_cstr);

    auto writes = stmt->writes();
    EXPECT_EQ(writes.size(), 1);
    auto write_access = *writes.begin();
    EXPECT_EQ(write_access->data(), "B");
    EXPECT_EQ(write_access->access_type(), analysis::AccessType::WRITE);
    const char* write_relation_cstr = isl_map_to_str(write_access->relation());
    std::string write_relation = write_relation_cstr;
    EXPECT_EQ(write_relation, "{ S_8[i] -> [0] }");
    free((void*) write_relation_cstr);

    isl_set_free(domain);
}

TEST(ScopAnalysisTest, ScopBuilderTest_LoopWithConstantBounds) {
    builder::StructuredSDFGBuilder builder("simple_loop_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer int_ptr(int_type);
    types::Pointer opaque_desc;
    builder.add_container("i", int_type);
    builder.add_container("A", opaque_desc);
    builder.add_container("B", opaque_desc);

    // Create loop: for i = 0 to 10
    auto i_sym = symbolic::symbol("i");
    auto n_val = symbolic::integer(10);
    auto init = symbolic::integer(0);
    // update: i + 1
    auto update = symbolic::add(i_sym, symbolic::integer(1));
    // condition: i < 10
    auto cond = symbolic::Lt(i_sym, n_val);

    auto& loop = builder.add_for(root, i_sym, cond, init, update);

    // Add body
    auto& block = builder.add_block(loop.root());

    // Add dummy tasklet
    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});
    builder.add_computational_memlet(block, in_node, tasklet, "_int", {i_sym}, int_ptr);
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {i_sym}, int_ptr);

    // Run Analysis
    analysis::AnalysisManager am(sdfg);

    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    auto statements = scop->statements();
    ASSERT_EQ(statements.size(), 1);

    auto* stmt = statements[0];
    isl_set* domain = stmt->domain();
    ASSERT_NE(domain, nullptr);

    // Verify properties of domain
    // It should be 1-dimensional
    EXPECT_EQ(isl_set_dim(domain, isl_dim_set), 1);

    // Verify name of dimension
    const char* dim_name = isl_set_get_dim_name(domain, isl_dim_set, 0);
    EXPECT_STREQ(dim_name, "i");

    // Verify bounds: i >= 0 and i < 10
    isl_ctx* ctx = scop->ctx();

    // Helper lambda for point checking
    auto check_point = [&](int val, bool expected) {
        isl_point* p = isl_point_zero(isl_set_get_space(domain));
        p = isl_point_set_coordinate_val(p, isl_dim_set, 0, isl_val_int_from_si(ctx, val));

        // Use isl_set_is_subset(from_point(p), domain)
        isl_set* p_set = isl_set_from_point(p);
        bool is_subset = isl_set_is_subset(p_set, domain);
        EXPECT_EQ(is_subset, expected) << "Value " << val << " expected " << expected;
        isl_set_free(p_set);
    };

    check_point(0, true);
    check_point(9, true);
    check_point(10, false);
    check_point(-1, false);

    // Verify memory accesses
    auto reads = stmt->reads();
    EXPECT_EQ(reads.size(), 1);
    auto read_access = *reads.begin();
    EXPECT_EQ(read_access->data(), "A");
    EXPECT_EQ(read_access->access_type(), analysis::AccessType::READ);
    const char* read_relation_cstr = isl_map_to_str(read_access->relation());
    std::string read_relation = read_relation_cstr;
    EXPECT_EQ(read_relation, "{ S_8[i] -> [i] }");
    free((void*) read_relation_cstr);

    auto writes = stmt->writes();
    EXPECT_EQ(writes.size(), 1);
    auto write_access = *writes.begin();
    EXPECT_EQ(write_access->data(), "B");
    EXPECT_EQ(write_access->access_type(), analysis::AccessType::WRITE);
    const char* write_relation_cstr = isl_map_to_str(write_access->relation());
    std::string write_relation = write_relation_cstr;
    EXPECT_EQ(write_relation, "{ S_8[i] -> [i] }");
    free((void*) write_relation_cstr);

    // Verify Schedule
    isl_union_map* schedule = scop->schedule();
    std::string expected_str = "{ " + stmt->name() + "[i] -> [i] }";
    isl_union_map* expected = isl_union_map_read_from_str(scop->ctx(), expected_str.c_str());
    EXPECT_TRUE(isl_union_map_is_equal(schedule, expected));
    isl_union_map_free(expected);

    // Verify AST
    std::string ast = scop->ast();
    EXPECT_TRUE(ast.find("for (int c0 = 0; c0 <= 9; c0 += 1)") != std::string::npos);
}

TEST(ScopAnalysisTest, ScopBuilderTest_Loop_2D) {
    builder::StructuredSDFGBuilder builder("simple_loop_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Array int_array(int_type, symbolic::integer(4));
    types::Pointer int_ptr(int_array);
    types::Pointer opaque_desc;
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);
    builder.add_container("A", opaque_desc);
    builder.add_container("B", opaque_desc);

    // Create loop: for i = 0 to 10
    auto i_sym = symbolic::symbol("i");
    auto n_val = symbolic::integer(10);
    auto init = symbolic::integer(0);
    // update: i + 1
    auto update = symbolic::add(i_sym, symbolic::integer(1));
    // condition: i < 10
    auto cond = symbolic::Lt(i_sym, n_val);

    auto& loop = builder.add_for(root, i_sym, cond, init, update);

    auto sym_j = symbolic::symbol("j");
    auto& loop2 = builder.add_for(
        loop.root(),
        sym_j,
        symbolic::Lt(sym_j, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(sym_j, symbolic::integer(1))
    );

    // Add body
    auto& block = builder.add_block(loop2.root());

    // Add dummy tasklet
    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});
    builder.add_computational_memlet(block, in_node, tasklet, "_int", {i_sym, sym_j}, int_ptr);
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {i_sym, sym_j}, int_ptr);

    // Run Analysis
    analysis::AnalysisManager am(sdfg);

    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    auto statements = scop->statements();
    ASSERT_EQ(statements.size(), 1);

    auto* stmt = statements[0];
    isl_set* domain = stmt->domain();
    ASSERT_NE(domain, nullptr);

    // Verify properties of domain
    // It should be 2-dimensional
    EXPECT_EQ(isl_set_dim(domain, isl_dim_set), 2);

    // Verify name of dimension
    const char* dim_name = isl_set_get_dim_name(domain, isl_dim_set, 0);
    EXPECT_STREQ(dim_name, "i");
    dim_name = isl_set_get_dim_name(domain, isl_dim_set, 1);
    EXPECT_STREQ(dim_name, "j");

    // Verify bounds: i >= 0 and i < 10
    isl_ctx* ctx = scop->ctx();

    // Helper lambda for point checking
    auto check_point = [&](int val, bool expected, int dim) {
        isl_point* p = isl_point_zero(isl_set_get_space(domain));
        p = isl_point_set_coordinate_val(p, isl_dim_set, dim, isl_val_int_from_si(ctx, val));

        // Use isl_set_is_subset(from_point(p), domain)
        isl_set* p_set = isl_set_from_point(p);
        bool is_subset = isl_set_is_subset(p_set, domain);
        EXPECT_EQ(is_subset, expected) << "Value " << val << " expected " << expected;
        isl_set_free(p_set);
    };

    check_point(0, true, 0);
    check_point(9, true, 0);
    check_point(10, false, 0);
    check_point(-1, false, 0);
    check_point(0, true, 1);
    check_point(3, true, 1);
    check_point(4, false, 1);
    check_point(-1, false, 1);

    // Verify memory accesses
    auto reads = stmt->reads();
    EXPECT_EQ(reads.size(), 1);
    auto read_access = *reads.begin();
    EXPECT_EQ(read_access->data(), "A");
    EXPECT_EQ(read_access->access_type(), analysis::AccessType::READ);
    const char* read_relation_cstr = isl_map_to_str(read_access->relation());
    std::string read_relation = read_relation_cstr;
    EXPECT_EQ(read_relation, "{ S_11[i, j] -> [i, j] }");
    free((void*) read_relation_cstr);

    auto writes = stmt->writes();
    EXPECT_EQ(writes.size(), 1);
    auto write_access = *writes.begin();
    EXPECT_EQ(write_access->data(), "B");
    EXPECT_EQ(write_access->access_type(), analysis::AccessType::WRITE);
    const char* write_relation_cstr = isl_map_to_str(write_access->relation());
    std::string write_relation = write_relation_cstr;
    EXPECT_EQ(write_relation, "{ S_11[i, j] -> [i, j] }");
    free((void*) write_relation_cstr);

    // Verify Schedule
    isl_union_map* schedule = scop->schedule();
    std::string expected_str = "{ " + stmt->name() + "[i, j] -> [i, j] }";
    isl_union_map* expected = isl_union_map_read_from_str(scop->ctx(), expected_str.c_str());
    EXPECT_TRUE(isl_union_map_is_equal(schedule, expected));
    isl_union_map_free(expected);

    // Verify AST
    std::string ast = scop->ast();
    EXPECT_TRUE(ast.find("for (int c0 = 0; c0 <= 9; c0 += 1)") != std::string::npos);
    EXPECT_TRUE(ast.find("for (int c1 = 0; c1 <= 3; c1 += 1)") != std::string::npos);
}

TEST(ScopAnalysisTest, ScopBuilderTest_Loop_2D_TriangularDomain) {
    builder::StructuredSDFGBuilder builder("simple_loop_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Array int_array(int_type, symbolic::integer(10));
    types::Pointer int_ptr(int_array);
    types::Pointer opaque_desc;
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);
    builder.add_container("A", opaque_desc);
    builder.add_container("B", opaque_desc);

    // Create loop: for i = 0 to 10
    auto i_sym = symbolic::symbol("i");
    auto n_val = symbolic::integer(10);
    auto init = symbolic::integer(0);
    // update: i + 1
    auto update = symbolic::add(i_sym, symbolic::integer(1));
    // condition: i < 10
    auto cond = symbolic::Lt(i_sym, n_val);

    auto& loop = builder.add_for(root, i_sym, cond, init, update);

    auto sym_j = symbolic::symbol("j");
    auto& loop2 = builder.add_for(
        loop.root(), sym_j, symbolic::Lt(sym_j, i_sym), symbolic::integer(0), symbolic::add(sym_j, symbolic::integer(1))
    );

    // Add body
    auto& block = builder.add_block(loop2.root());

    // Add dummy tasklet
    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});
    builder.add_computational_memlet(block, in_node, tasklet, "_int", {i_sym, sym_j}, int_ptr);
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {i_sym, sym_j}, int_ptr);

    // Run Analysis
    analysis::AnalysisManager am(sdfg);

    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    auto statements = scop->statements();
    ASSERT_EQ(statements.size(), 1);

    auto* stmt = statements[0];
    isl_set* domain = stmt->domain();
    ASSERT_NE(domain, nullptr);

    // Verify properties of domain
    // It should be 2-dimensional
    EXPECT_EQ(isl_set_dim(domain, isl_dim_set), 2);

    // Verify name of dimension
    const char* dim_name = isl_set_get_dim_name(domain, isl_dim_set, 0);
    EXPECT_STREQ(dim_name, "i");
    dim_name = isl_set_get_dim_name(domain, isl_dim_set, 1);
    EXPECT_STREQ(dim_name, "j");


    // Verify memory accesses
    auto reads = stmt->reads();
    EXPECT_EQ(reads.size(), 1);
    auto read_access = *reads.begin();
    EXPECT_EQ(read_access->data(), "A");
    EXPECT_EQ(read_access->access_type(), analysis::AccessType::READ);
    const char* read_relation_cstr = isl_map_to_str(read_access->relation());
    std::string read_relation = read_relation_cstr;
    EXPECT_EQ(read_relation, "{ S_11[i, j] -> [i, j] }");
    free((void*) read_relation_cstr);

    auto writes = stmt->writes();
    EXPECT_EQ(writes.size(), 1);
    auto write_access = *writes.begin();
    EXPECT_EQ(write_access->data(), "B");
    EXPECT_EQ(write_access->access_type(), analysis::AccessType::WRITE);
    const char* write_relation_cstr = isl_map_to_str(write_access->relation());
    std::string write_relation = write_relation_cstr;
    EXPECT_EQ(write_relation, "{ S_11[i, j] -> [i, j] }");
    free((void*) write_relation_cstr);

    // Verify Schedule
    isl_union_map* schedule = scop->schedule();
    std::string expected_str = "{ " + stmt->name() + "[i, j] -> [i, j] }";
    isl_union_map* expected = isl_union_map_read_from_str(scop->ctx(), expected_str.c_str());
    EXPECT_TRUE(isl_union_map_is_equal(schedule, expected));
    isl_union_map_free(expected);

    // Verify AST
    std::string ast = scop->ast();
    // ISL might optimize out the first iteration (c0=0) because the inner loop (j<i) is empty for i=0
    bool found_0 = ast.find("for (int c0 = 0; c0 <= 9; c0 += 1)") != std::string::npos;
    bool found_1 = ast.find("for (int c0 = 1; c0 <= 9; c0 += 1)") != std::string::npos;
    EXPECT_TRUE(found_0 || found_1);
    EXPECT_TRUE(ast.find("for (int c1 = 0; c1 < c0; c1 += 1)") != std::string::npos);
}

TEST(ScopAnalysisTest, ScopBuilderTest_LoopWithSymbolicBounds) {
    builder::StructuredSDFGBuilder builder("simple_loop_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer int_ptr(int_type);
    types::Pointer opaque_desc;
    builder.add_container("i", int_type);
    builder.add_container("N", int_type);
    builder.add_container("A", opaque_desc);
    builder.add_container("B", opaque_desc);

    // Create loop: for i = 0 to N
    auto i_sym = symbolic::symbol("i");
    auto n_val = symbolic::symbol("N");
    auto init = symbolic::integer(0);
    // update: i + 1
    auto update = symbolic::add(i_sym, symbolic::integer(1));
    // condition: i < N
    auto cond = symbolic::Lt(i_sym, n_val);

    auto& loop = builder.add_for(root, i_sym, cond, init, update);

    // Add body
    auto& block = builder.add_block(loop.root());

    // Add dummy tasklet
    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});
    builder.add_computational_memlet(block, in_node, tasklet, "_int", {i_sym}, int_ptr);
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {i_sym}, int_ptr);

    // Run Analysis
    analysis::AnalysisManager am(sdfg);

    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    auto statements = scop->statements();
    ASSERT_EQ(statements.size(), 1);

    auto* stmt = statements[0];
    isl_set* domain = stmt->domain();
    ASSERT_NE(domain, nullptr);

    // Verify properties of domain
    // It should be 1-dimensional
    EXPECT_EQ(isl_set_dim(domain, isl_dim_set), 1);

    // Verify name of dimension
    const char* dim_name = isl_set_get_dim_name(domain, isl_dim_set, 0);
    EXPECT_STREQ(dim_name, "i");

    // Verify memory accesses
    auto reads = stmt->reads();
    EXPECT_EQ(reads.size(), 1);
    auto read_access = *reads.begin();
    EXPECT_EQ(read_access->data(), "A");
    EXPECT_EQ(read_access->access_type(), analysis::AccessType::READ);
    const char* read_relation_cstr = isl_map_to_str(read_access->relation());
    std::string read_relation = read_relation_cstr;
    EXPECT_EQ(read_relation, "[N] -> { S_8[i] -> [i] }");
    free((void*) read_relation_cstr);

    auto writes = stmt->writes();
    EXPECT_EQ(writes.size(), 1);
    auto write_access = *writes.begin();
    EXPECT_EQ(write_access->data(), "B");
    EXPECT_EQ(write_access->access_type(), analysis::AccessType::WRITE);
    const char* write_relation_cstr = isl_map_to_str(write_access->relation());
    std::string write_relation = write_relation_cstr;
    EXPECT_EQ(write_relation, "[N] -> { S_8[i] -> [i] }");
    free((void*) write_relation_cstr);
}

TEST(ScopAnalysisTest, ScopBuilderTest_LoopWithStrides) {
    builder::StructuredSDFGBuilder builder("simple_loop_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer int_ptr(int_type);
    types::Pointer opaque_desc;
    builder.add_container("i", int_type);
    builder.add_container("N", int_type);
    builder.add_container("A", opaque_desc);
    builder.add_container("B", opaque_desc);

    // Create loop: for i = 0 to N
    auto i_sym = symbolic::symbol("i");
    auto n_val = symbolic::symbol("N");
    auto init = symbolic::integer(0);
    // update: i + 1
    auto update = symbolic::add(i_sym, symbolic::integer(2));
    // condition: i < N
    auto cond = symbolic::Lt(i_sym, n_val);

    auto& loop = builder.add_for(root, i_sym, cond, init, update);

    // Add body
    auto& block = builder.add_block(loop.root());

    // Add dummy tasklet
    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});
    builder.add_computational_memlet(block, in_node, tasklet, "_int", {i_sym}, int_ptr);
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {i_sym}, int_ptr);

    // Run Analysis
    analysis::AnalysisManager am(sdfg);

    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    auto statements = scop->statements();
    ASSERT_EQ(statements.size(), 1);

    auto* stmt = statements[0];
    isl_set* domain = stmt->domain();
    ASSERT_NE(domain, nullptr);

    // Verify properties of domain
    // It should be 1-dimensional
    EXPECT_EQ(isl_set_dim(domain, isl_dim_set), 1);

    // Verify name of dimension
    const char* dim_name = isl_set_get_dim_name(domain, isl_dim_set, 0);
    EXPECT_STREQ(dim_name, "i");

    // Verify memory accesses
    auto reads = stmt->reads();
    EXPECT_EQ(reads.size(), 1);
    auto read_access = *reads.begin();
    EXPECT_EQ(read_access->data(), "A");
    EXPECT_EQ(read_access->access_type(), analysis::AccessType::READ);
    const char* read_relation_cstr = isl_map_to_str(read_access->relation());
    std::string read_relation = read_relation_cstr;
    EXPECT_EQ(read_relation, "[N] -> { S_8[i] -> [i] }");
    free((void*) read_relation_cstr);

    auto writes = stmt->writes();
    EXPECT_EQ(writes.size(), 1);
    auto write_access = *writes.begin();
    EXPECT_EQ(write_access->data(), "B");
    EXPECT_EQ(write_access->access_type(), analysis::AccessType::WRITE);
    const char* write_relation_cstr = isl_map_to_str(write_access->relation());
    std::string write_relation = write_relation_cstr;
    EXPECT_EQ(write_relation, "[N] -> { S_8[i] -> [i] }");
    free((void*) write_relation_cstr);

    std::string ast = scop->ast();
    EXPECT_TRUE(ast.find("for (int c0 = 0; c0 < N; c0 += 2)") != std::string::npos);
}

TEST(ScopAnalysisTest, ScopBuilderTest_Loop_2D_Symbolic) {
    builder::StructuredSDFGBuilder builder("simple_loop_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Array int_array(int_type, symbolic::integer(4));
    types::Pointer int_ptr(int_array);
    types::Pointer opaque_desc;
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);
    builder.add_container("N", int_type);
    builder.add_container("M", int_type);
    builder.add_container("A", opaque_desc);
    builder.add_container("B", opaque_desc);

    // Create loop: for i = 0 to 10
    auto i_sym = symbolic::symbol("i");
    auto n_val = symbolic::symbol("N");
    auto init = symbolic::integer(0);
    // update: i + 1
    auto update = symbolic::add(i_sym, symbolic::integer(1));
    // condition: i < 10
    auto cond = symbolic::Lt(i_sym, n_val);

    auto& loop = builder.add_for(root, i_sym, cond, init, update);

    auto sym_j = symbolic::symbol("j");
    auto& loop2 = builder.add_for(
        loop.root(),
        sym_j,
        symbolic::Lt(sym_j, symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(sym_j, symbolic::integer(1))
    );

    // Add body
    auto& block = builder.add_block(loop2.root());

    // Add dummy tasklet
    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});
    builder.add_computational_memlet(block, in_node, tasklet, "_int", {i_sym, sym_j}, int_ptr);
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {i_sym, sym_j}, int_ptr);

    // Run Analysis
    analysis::AnalysisManager am(sdfg);

    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    auto statements = scop->statements();
    ASSERT_EQ(statements.size(), 1);

    auto* stmt = statements[0];
    isl_set* domain = stmt->domain();
    ASSERT_NE(domain, nullptr);

    // Verify properties of domain
    // It should be 2-dimensional
    EXPECT_EQ(isl_set_dim(domain, isl_dim_set), 2);

    // Verify name of dimension
    const char* dim_name = isl_set_get_dim_name(domain, isl_dim_set, 0);
    EXPECT_STREQ(dim_name, "i");
    dim_name = isl_set_get_dim_name(domain, isl_dim_set, 1);
    EXPECT_STREQ(dim_name, "j");

    // Verify memory accesses
    auto reads = stmt->reads();
    EXPECT_EQ(reads.size(), 1);
    auto read_access = *reads.begin();
    EXPECT_EQ(read_access->data(), "A");
    EXPECT_EQ(read_access->access_type(), analysis::AccessType::READ);
    const char* read_relation_cstr = isl_map_to_str(read_access->relation());
    std::string read_relation = read_relation_cstr;
    EXPECT_EQ(read_relation, "[N, M] -> { S_11[i, j] -> [i, j] }");
    free((void*) read_relation_cstr);

    auto writes = stmt->writes();
    EXPECT_EQ(writes.size(), 1);
    auto write_access = *writes.begin();
    EXPECT_EQ(write_access->data(), "B");
    EXPECT_EQ(write_access->access_type(), analysis::AccessType::WRITE);
    const char* write_relation_cstr = isl_map_to_str(write_access->relation());
    std::string write_relation = write_relation_cstr;
    EXPECT_EQ(write_relation, "[N, M] -> { S_11[i, j] -> [i, j] }");
    free((void*) write_relation_cstr);

    // Verify Schedule
    isl_union_map* schedule = scop->schedule();
    std::string expected_str = "{ " + stmt->name() + "[i, j] -> [i, j] }";
    isl_union_map* expected = isl_union_map_read_from_str(scop->ctx(), expected_str.c_str());
    EXPECT_TRUE(isl_union_map_is_equal(schedule, expected));
    isl_union_map_free(expected);

    // Verify AST
    std::string ast = scop->ast();
    EXPECT_TRUE(ast.find("for (int c0 = 0; c0 < N; c0 += 1)") != std::string::npos);
    EXPECT_TRUE(ast.find("for (int c1 = 0; c1 < M; c1 += 1)") != std::string::npos);
}

TEST(ScopAnalysisTest, ScopBuilderTest_NonPerfectlyNestedLoop) {
    builder::StructuredSDFGBuilder builder("non_perfectly_nested", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer ptr_type(int_type);
    types::Array int_array(int_type, symbolic::integer(10));
    types::Pointer int_ptr(int_array);
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);
    builder.add_container("A", ptr_type);
    builder.add_container("B", int_ptr);

    // Create loop: for i = 0 to 10
    auto i_sym = symbolic::symbol("i");
    auto n_val = symbolic::integer(10);
    auto init = symbolic::integer(0);
    // update: i + 1
    auto update = symbolic::add(i_sym, symbolic::integer(1));
    // condition: i < 10
    auto cond = symbolic::Lt(i_sym, n_val);

    auto& loop = builder.add_for(root, i_sym, cond, init, update);

    // Add S1 (initialization of A) inside outer loop
    auto& block1 = builder.add_block(loop.root());
    auto& A_out = builder.add_access(block1, "A");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", A_out, {i_sym}, ptr_type);

    // Inner loop: for j = 0 to i
    auto j_sym = symbolic::symbol("j");
    auto& loop2 = builder.add_for(
        loop.root(), j_sym, symbolic::Lt(j_sym, i_sym), symbolic::integer(0), symbolic::add(j_sym, symbolic::integer(1))
    );

    // Add S2 inside inner loop
    auto& block2 = builder.add_block(loop2.root());
    auto& A_in = builder.add_access(block2, "A");
    auto& B_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, A_in, tasklet2, "_in", {i_sym}, ptr_type);
    builder.add_computational_memlet(block2, tasklet2, "_out", B_out, {i_sym, j_sym}, int_ptr);

    // Run Analysis
    analysis::AnalysisManager am(sdfg);

    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    auto statements = scop->statements();
    ASSERT_EQ(statements.size(), 2);

    // Find statements
    analysis::ScopStatement* s1 = nullptr;
    analysis::ScopStatement* s2 = nullptr;

    for (auto* stmt : statements) {
        int dim = isl_set_dim(stmt->domain(), isl_dim_set);
        if (dim == 1) {
            s1 = stmt;
        } else if (dim == 2) {
            s2 = stmt;
        }
    }

    ASSERT_NE(s1, nullptr) << "Could not find 1D statement (S1)";
    ASSERT_NE(s2, nullptr) << "Could not find 2D statement (S2)";

    // Verify S1 domain: 0 <= i < 10
    auto d1 = s1->domain();
    EXPECT_EQ(isl_set_dim(d1, isl_dim_set), 1);
    EXPECT_STREQ(isl_set_get_dim_name(d1, isl_dim_set, 0), "i");

    // Verify S2 domain: 0 <= i < 10 and 0 <= j < i
    auto d2 = s2->domain();
    EXPECT_EQ(isl_set_dim(d2, isl_dim_set), 2);
    EXPECT_STREQ(isl_set_get_dim_name(d2, isl_dim_set, 0), "i");
    EXPECT_STREQ(isl_set_get_dim_name(d2, isl_dim_set, 1), "j");

    // Check if AST generation works (implies schedule validity)
    std::string ast = scop->ast();
    // Verify AST structure contains Outer loop
    EXPECT_TRUE(ast.find("S_16(c0, c1);") != std::string::npos);
    EXPECT_TRUE(ast.find("S_7(c0);") != std::string::npos);
}

TEST(ScopAnalysisTest, DependenceInfoTest_RAW_Dependence) {
    builder::StructuredSDFGBuilder builder("raw_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer int_ptr(int_type);
    builder.add_container("A", int_ptr);
    builder.add_container("B", int_ptr);

    // Create loop: for i = 1 to 10
    auto i_sym = symbolic::symbol("i");
    auto start = symbolic::integer(1);
    auto end = symbolic::integer(11); // Loop upper bound is exclusive usually? add_for cond usually '<'.
    // Cond: i < 11
    auto cond = symbolic::Lt(i_sym, end);
    auto update = symbolic::add(i_sym, symbolic::integer(1));

    auto& loop = builder.add_for(root, i_sym, cond, start, update);

    // A[i] = A[i-1]
    auto& block = builder.add_block(loop.root());

    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});

    // Read A[i-1]
    builder
        .add_computational_memlet(block, in_node, tasklet, "_int", {symbolic::sub(i_sym, symbolic::integer(1))}, int_ptr);
    // Write A[i]
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {i_sym}, int_ptr);

    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop); // passed loop as root because scop is built for the loop
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);

    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);

    isl_union_map* raw = deps.dependences(analysis::Dependences::TYPE_RAW);
    ASSERT_FALSE(isl_union_map_is_empty(raw));

    // Expected: { Stmt[i] -> Stmt[i+1] : 1 <= i <= 9 }
    // Note: Stmt names might be generated. Let's get the statement instance.
    auto stmts = scop->statements();
    ASSERT_EQ(stmts.size(), 1);
    // Since we don't know the statement name easily (it's generated), we can iterate or build expected string with
    // wildcard or verify using subset.

    // Let's print for debugging if test fails (but here I'm writing code)
    // Checking if Stmt[1] -> Stmt[2] exists.

    // Construct domain point for Statement at i=1
    // And range point for Statement at i=2

    // Actually, easier to check if the map is exactly { S[i] -> S[i+1] : 1 <= i <= 9 }
    // Get the statement id/name
    isl_set* domain = stmts[0]->domain();
    const char* stmt_name = isl_set_get_tuple_name(domain);

    std::string expected_str = "{ ";
    expected_str += stmt_name;
    expected_str += "[i] -> ";
    expected_str += stmt_name;
    expected_str += "[i + 1] : 1 <= i <= 9 }";

    isl_union_map* expected = isl_union_map_read_from_str(scop->ctx(), expected_str.c_str());

    // Intersect with calculated RAW to see if it matches
    // But RAW might contain more info or have different domain constraints if I messed up loop bounds.
    // Loop 1 to 10 inclusive (i < 11).
    // i=1: read A[0], write A[1].
    // i=2: read A[1], write A[2]. raw dep on A[1] from i=1.
    // ...
    // i=10: read A[9], write A[10]. raw dep on A[9] from i=9.
    // So source i=1..9. Sink i=2..10.
    // S[i] -> S[i+1] for i in 1..9.

    // The Dependences class returns both tagged (fine-grained) and statement-level dependences.
    // We verify that our expected statement-level dependence is present.
    bool isSubset = isl_union_map_is_subset(expected, raw);
    if (!isSubset) {
        std::cout << "Computed RAW: " << isl_union_map_to_str(raw) << std::endl;
        std::cout << "Expected RAW: " << isl_union_map_to_str(expected) << std::endl;
    }
    EXPECT_TRUE(isSubset);

    isl_union_map_free(raw);
    isl_union_map_free(expected);
}

TEST(ScopAnalysisTest, DependenceInfoTest_WAR_Dependence) {
    builder::StructuredSDFGBuilder builder("war_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer int_ptr(int_type);
    builder.add_container("A", int_ptr);

    auto i_sym = symbolic::symbol("i");
    auto start = symbolic::integer(0);
    auto end = symbolic::integer(10);
    auto cond = symbolic::Lt(i_sym, end);
    auto update = symbolic::add(i_sym, symbolic::integer(1));

    auto& loop = builder.add_for(root, i_sym, cond, start, update);

    // A[i] = A[i+1]
    auto& block = builder.add_block(loop.root());

    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});

    // Read A[i+1]
    builder
        .add_computational_memlet(block, in_node, tasklet, "_int", {symbolic::add(i_sym, symbolic::integer(1))}, int_ptr);
    // Write A[i]
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {i_sym}, int_ptr);

    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);

    isl_union_map* war = deps.dependences(analysis::Dependences::TYPE_WAR);

    // i=0: Read A[1], Write A[0].
    // i=1: Read A[2], Write A[1]. write A[1] WAR depend on read A[1] at i=0.
    // Source: Read at i=0. Sink: Write at i=1.
    // Dep: S[0] -> S[1].
    // generally S[i] -> S[i+1].
    // Range i: 0 to 9.
    // source i=0..8. Sink i=1..9.

    isl_set* domain = scop->statements()[0]->domain();
    const char* stmt_name = isl_set_get_tuple_name(domain);

    std::string expected_str = "{ ";
    expected_str += stmt_name;
    expected_str += "[i] -> ";
    expected_str += stmt_name;
    expected_str += "[i + 1] : 0 <= i <= 8 }";

    isl_union_map* expected = isl_union_map_read_from_str(scop->ctx(), expected_str.c_str());

    // Check subset (to ignore tagged deps)
    bool isSubset = isl_union_map_is_subset(expected, war);
    if (!isSubset) {
        std::cout << "Computed WAR: " << isl_union_map_to_str(war) << std::endl;
        std::cout << "Expected WAR: " << isl_union_map_to_str(expected) << std::endl;
    }
    EXPECT_TRUE(isSubset);

    isl_union_map_free(war);
    isl_union_map_free(expected);
}

TEST(ScopAnalysisTest, DependenceInfoTest_WAW_Dependence) {
    // For WAW, maybe 2 statements writing to same location?
    // Or same statement: A[0] = ... inside a loop.

    builder::StructuredSDFGBuilder builder("waw_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer int_ptr(int_type);
    builder.add_container("A", int_ptr);

    auto i_sym = symbolic::symbol("i");
    auto start = symbolic::integer(0);
    auto end = symbolic::integer(10);
    auto cond = symbolic::Lt(i_sym, end);
    auto update = symbolic::add(i_sym, symbolic::integer(1));

    auto& loop = builder.add_for(root, i_sym, cond, start, update);

    // A[0] = i
    auto& block = builder.add_block(loop.root());

    auto& val_node = builder.add_access(block, "i"); // reading loop var?
    // Wait, "i" is a container created by add_for if I added it as container?
    // In RAW test I didn't add "i" as container explicitly, but maybe "A" memlet used symbol directly.
    // symbol usage in memlet index is fine. But tasklet input needs a memlet from a node.

    // Let's just do A[0] = 5;
    // Actually simpler: just write to A[0].

    auto& out_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {});

    // write A[0]
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {symbolic::integer(0)}, int_ptr);

    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);

    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);

    isl_union_map* waw = deps.dependences(analysis::Dependences::TYPE_WAW);

    // i=0: Write A[0]
    // i=1: Write A[0] -> WAW on i=0
    // ...
    // S[i] -> S[i+1]

    isl_set* domain = scop->statements()[0]->domain();
    const char* stmt_name = isl_set_get_tuple_name(domain);

    std::string expected_str = "{ ";
    expected_str += stmt_name;
    expected_str += "[i] -> ";
    expected_str += stmt_name;
    expected_str += "[i + 1] : 0 <= i <= 8 }";
    // 0 to 9. i=8 writes -> i=9 writes.

    isl_union_map* expected = isl_union_map_read_from_str(scop->ctx(), expected_str.c_str());
    bool isSubset = isl_union_map_is_subset(expected, waw);
    if (!isSubset) {
        std::cout << "Computed WAW: " << isl_union_map_to_str(waw) << std::endl;
        std::cout << "Expected WAW: " << isl_union_map_to_str(expected) << std::endl;
    }
    EXPECT_TRUE(isSubset);

    isl_union_map_free(waw);
    isl_union_map_free(expected);
}

// Validity Check Test
TEST(ScopAnalysisTest, DependenceInfoTest_Validity) {
    // RAW S[i] -> S[i+1].
    // Schedule i -> i is valid.
    // Schedule i -> -i is invalid (reversed). S[i+1] executes before S[i].

    builder::StructuredSDFGBuilder builder("validity_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer int_ptr(int_type);
    builder.add_container("A", int_ptr);

    auto i_sym = symbolic::symbol("i");
    auto start = symbolic::integer(1);
    auto end = symbolic::integer(10);
    auto cond = symbolic::Lt(i_sym, end);
    auto update = symbolic::add(i_sym, symbolic::integer(1));
    auto& loop = builder.add_for(root, i_sym, cond, start, update);
    auto& block = builder.add_block(loop.root());
    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});
    // A[i] = A[i-1] -> RAW S[i-1] -> S[i]
    builder
        .add_computational_memlet(block, in_node, tasklet, "_int", {symbolic::sub(i_sym, symbolic::integer(1))}, int_ptr);
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {i_sym}, int_ptr);

    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);

    // Default schedule should be valid
    EXPECT_TRUE(deps.is_valid(*scop, scop->schedule_tree()));

    // Create reversed schedule: i -> -i
    // We can create a new schedule map
    // { S[i] -> [-i] }

    isl_set* domain = scop->statements()[0]->domain();
    const char* stmt_name = isl_set_get_tuple_name(domain);
    std::string stmt_name_str = stmt_name;
    std::string sched_str = "{ ";
    sched_str += stmt_name_str;
    sched_str += "[i] -> [-i] }";

    isl_map* new_sched_map = isl_map_read_from_str(scop->ctx(), sched_str.c_str());
    new_sched_map = isl_map_intersect_domain(new_sched_map, domain);

    std::unordered_map<analysis::ScopStatement*, isl_map*> new_schedule;
    new_schedule[scop->statements()[0]] = new_sched_map;

    EXPECT_FALSE(deps.is_valid(*scop, new_schedule));

    isl_map_free(new_sched_map);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Last_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", base_desc, true);

    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a1 = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a1, tasklet, "_in", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("B"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Sum_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", base_desc, true);

    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a1 = builder.add_access(block, "A");
    auto& b_in = builder.add_access(block, "B");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a1, tasklet, "_in1", {indvar}, edge_desc);
    builder.add_computational_memlet(block, b_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {});

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("B"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Shift_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a1 = builder.add_access(block, "A");
    auto& a2 = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, a1, tasklet, "_in", {symbolic::sub(indvar, symbolic::integer(1))}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a2, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(ScopAnalysisTest, DependenceInfoTest_PartialSum_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A1 = builder.add_access(block, "A");
    auto& A2 = builder.add_access(block, "A");
    auto& A3 = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, A1, tasklet, "_in1", {symbolic::sub(indvar, symbolic::integer(1))}, edge_desc);
    builder.add_computational_memlet(block, A2, tasklet, "_in2", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A3, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(ScopAnalysisTest, DependenceInfoTest_LoopLocal_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("tmp", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block_1 = builder.add_block(body);
    auto& i_in = builder.add_access(block_1, "i");
    auto& tmp_out = builder.add_access(block_1, "tmp");
    auto& tasklet_1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_1, i_in, tasklet_1, "_in", {});
    builder.add_computational_memlet(block_1, tasklet_1, "_out", tmp_out, {});

    auto& block_2 = builder.add_block(body);
    auto& tmp_in = builder.add_access(block_2, "tmp");
    auto& a_out = builder.add_access(block_2, "A");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_2, tmp_in, tasklet, "_in", {});
    builder.add_computational_memlet(block_2, tasklet, "_out", a_out, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("tmp"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(ScopAnalysisTest, DISABLED_DependenceInfoTest_LoopLocal_Conditional) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("i", sym_desc);
    builder.add_container("tmp", sym_desc);

    builder.add_container("A", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto& ifelse = builder.add_if_else(body1);
    auto& branch1 = builder.add_case(ifelse, symbolic::Eq(indvar1, symbolic::integer(0)));
    auto& block1 = builder.add_block(branch1, {{symbolic::symbol("tmp"), symbolic::zero()}});
    auto& branch2 = builder.add_case(ifelse, symbolic::Ne(indvar1, symbolic::integer(0)));
    auto& block2 = builder.add_block(branch2, {{symbolic::symbol("tmp"), symbolic::one()}});

    // Add computation
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "tmp");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop1);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop1);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("tmp"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(ScopAnalysisTest, DISABLED_DependenceInfoTest_LoopLocal_Conditional_Incomplete) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("i", sym_desc);
    builder.add_container("tmp", sym_desc);
    builder.add_container("A", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto& ifelse = builder.add_if_else(body1);
    auto& branch1 = builder.add_case(ifelse, symbolic::Eq(indvar1, symbolic::integer(0)));
    auto& block1 = builder.add_block(branch1, {{symbolic::symbol("tmp"), symbolic::zero()}});

    // Add computation
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "tmp");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop1);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop1);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("tmp"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Store_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block, tasklet, "_out", a, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Copy_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Map_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Map_1D_Disjoint) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::zero()}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::add(symbolic::symbol("i"), symbolic::one())}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Map_1D_Strided) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::sub(symbolic::symbol("N"), symbolic::integer(1));
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::add(symbolic::symbol("i"), symbolic::one())}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Map_1D_Strided2) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(
        block, tasklet, "_out", a_out, {symbolic::sub(symbolic::symbol("i"), symbolic::one())}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Map_1D_Tiled) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("i_tile", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i_tile");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto tile_size = symbolic::integer(8);
    auto update = symbolic::add(indvar, tile_size);

    auto& loop_outer = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop_outer.root();

    auto indvar_tile = symbolic::symbol("i");
    auto init_tile = indvar;
    auto condition_tile = symbolic::
        And(symbolic::Lt(indvar_tile, symbolic::symbol("N")),
            symbolic::Lt(indvar_tile, symbolic::add(indvar, tile_size)));
    auto update_tile = symbolic::add(indvar_tile, symbolic::one());

    auto& loop_inner = builder.add_for(body, indvar_tile, condition_tile, init_tile, update_tile);
    auto& body_inner = loop_inner.root();

    // Add computation
    auto& block = builder.add_block(body_inner);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop_outer);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies1 = deps.dependencies(loop_outer);
    auto dependencies2 = deps.dependencies(loop_inner);

    // Check
    EXPECT_EQ(dependencies1.size(), 0);
    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(ScopAnalysisTest, DISABLED_DependenceInfoTest_Map_1D_Incomplete) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar bool_desc(types::PrimitiveType::Bool);
    types::Scalar sym_desc(types::PrimitiveType::Int32);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("tmp", base_desc);
    builder.add_container("k", bool_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // tmp = A[i]
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& tmp_out = builder.add_access(block, "tmp");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", tmp_out, {});

    // switch = tmp > 0
    auto& block_switch = builder.add_block(body);
    auto& tmp_in = builder.add_access(block_switch, "tmp");
    auto& zero_node = builder.add_constant(block_switch, "0.0", base_desc);
    auto& switch_out = builder.add_access(block_switch, "k");
    auto& tasklet_switch = builder.add_tasklet(block_switch, data_flow::TaskletCode::fp_oge, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block_switch, tmp_in, tasklet_switch, "_in1", {});
    builder.add_computational_memlet(block_switch, zero_node, tasklet_switch, "_in2", {});
    builder.add_computational_memlet(block_switch, tasklet_switch, "_out", switch_out, {});

    auto switch_condition = symbolic::Eq(symbolic::symbol("k"), symbolic::__true__());
    auto& ifelse = builder.add_if_else(body);

    // if (switch) A[i] = tmp
    auto& branch1 = builder.add_case(ifelse, switch_condition);
    auto& block1 = builder.add_block(branch1);
    auto& tmp_in1 = builder.add_access(block1, "tmp");
    auto& a_out1 = builder.add_access(block1, "A");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, tmp_in1, tasklet1, "_in", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", a_out1, {indvar}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 2);
    EXPECT_EQ(dependencies.at("tmp"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies.at("k"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(ScopAnalysisTest, DISABLED_DependenceInfoTest_MapParameterized_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("m", sym_desc, true);
    builder.add_container("b", sym_desc, true);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in1"});
    builder.add_computational_memlet(
        block,
        A_in,
        tasklet,
        "_in1",
        {symbolic::add(symbolic::mul(symbolic::symbol("m"), symbolic::symbol("i")), symbolic::symbol("b"))},
        edge_desc
    );
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        A_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("m"), symbolic::symbol("i")), symbolic::symbol("b"))},
        edge_desc
    );

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    // m == 0 -> all iterations access the same location
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Stencil_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(1);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A1 = builder.add_access(block, "A");
    auto& A2 = builder.add_access(block, "A");
    auto& A3 = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(
        block, A1, tasklet, "_in1", {symbolic::sub(symbolic::symbol("i"), symbolic::integer(1))}, edge_desc
    );
    builder.add_computational_memlet(block, A2, tasklet, "_in2", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(
        block, A3, tasklet, "_in3", {symbolic::add(symbolic::symbol("i"), symbolic::integer(1))}, edge_desc
    );
    builder.add_computational_memlet(block, tasklet, "_out", B, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 0);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Gather_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("b", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block_1 = builder.add_block(body);

    // Define indirection
    auto& A = builder.add_access(block_1, "A");
    auto& b = builder.add_access(block_1, "b");
    auto& tasklet1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_1, A, tasklet1, "_in", {indvar}, edge_desc);
    builder.add_computational_memlet(block_1, tasklet1, "_out", b, {});

    auto& block_2 = builder.add_block(body);
    auto& B = builder.add_access(block_2, "B");
    auto& C = builder.add_access(block_2, "C");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_2, B, tasklet, "_in", {symbolic::symbol("b")}, edge_desc);
    builder.add_computational_memlet(block_2, tasklet, "_out", C, {symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 1);
    EXPECT_EQ(dependencies.at("b"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Scatter_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Int64);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);
    builder.add_container("C", desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("b", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Define indirection
    auto& block_1 = builder.add_block(body);
    auto& A = builder.add_access(block_1, "A");
    auto& b = builder.add_access(block_1, "b");
    auto& tasklet1 = builder.add_tasklet(block_1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_1, A, tasklet1, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block_1, tasklet1, "_out", b, {});

    auto& block_2 = builder.add_block(body);
    auto& B = builder.add_access(block_2, "B");
    auto& C = builder.add_access(block_2, "C");
    auto& tasklet = builder.add_tasklet(block_2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_2, B, tasklet, "_in", {symbolic::symbol("i")}, edge_desc);
    builder.add_computational_memlet(block_2, tasklet, "_out", C, {symbolic::symbol("b")}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    // Check
    EXPECT_EQ(dependencies.size(), 2);
    EXPECT_EQ(dependencies.at("b"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
    EXPECT_EQ(dependencies.at("C"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE);
}

TEST(ScopAnalysisTest, DISABLED_DependenceInfoTest_MapDeg2_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer edge_desc(base_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(
        block, tasklet, "_out", A, {symbolic::mul(symbolic::symbol("i"), symbolic::symbol("i"))}, edge_desc
    );

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_EQ(scop, nullptr);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Map_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("M"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop 1
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Define loop 2
    auto bound_2 = symbolic::symbol("M");
    auto indvar_2 = symbolic::symbol("j");
    auto init_2 = symbolic::integer(0);
    auto condition_2 = symbolic::Lt(indvar_2, bound_2);
    auto update_2 = symbolic::add(indvar_2, symbolic::integer(1));

    auto& loop_2 = builder.add_for(body, indvar_2, condition_2, init_2, update_2);
    auto& body_2 = loop_2.root();

    // Add computation
    auto& block = builder.add_block(body_2);
    auto& a_in = builder.add_access(block, "A");
    auto& one_node = builder.add_constant(block, "1.0", base_desc);
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder.add_computational_memlet(block, one_node, tasklet, "_in2", {});
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies = deps.dependencies(loop);

    EXPECT_EQ(dependencies.size(), 0);

    // Check loop 2
    auto dependencies_2 = deps.dependencies(loop_2);
    EXPECT_EQ(dependencies_2.size(), 0);
}

TEST(ScopAnalysisTest, DependenceInfoTest_PartialSumInner_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", array_desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("N");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A = builder.add_access(block, "A");
    auto& B1 = builder.add_access(block, "B");
    auto& B2 = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block, A, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder.add_computational_memlet(block, B1, tasklet, "_in2", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block, tasklet, "_out", B2, {symbolic::symbol("i")}, array_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop1);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies1 = deps.dependencies(loop1);
    auto dependencies2 = deps.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 0);

    EXPECT_EQ(dependencies2.size(), 1);
    EXPECT_EQ(dependencies2.at("B"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(ScopAnalysisTest, DISABLED_DependenceInfoTest_PartialSumOuter_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", desc, true);
    builder.add_container("B", array_desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("N");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A = builder.add_access(block, "A");
    auto& B1 = builder.add_access(block, "B");
    auto& B2 = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A, tasklet, "_in1", {indvar1, indvar2}, edge_desc);
    builder.add_computational_memlet(block, B1, tasklet, "_in2", {indvar2}, array_desc);
    builder.add_computational_memlet(block, tasklet, "_out", B2, {indvar2}, array_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop1);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies1 = deps.dependencies(loop1);
    auto dependencies2 = deps.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("B"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);

    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(ScopAnalysisTest, DependenceInfoTest_PartialSum_1D_Triangle) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(base_desc);

    types::Pointer desc;
    builder.add_container("A", desc, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    // init block
    auto& init_block = builder.add_block(body1);
    auto& A_init = builder.add_access(init_block, "A");
    auto& zero_node = builder.add_constant(init_block, "0.0", base_desc);
    auto& tasklet_init = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(init_block, zero_node, tasklet_init, "_in", {});
    builder.add_computational_memlet(init_block, tasklet_init, "_out", A_init, {indvar1}, ptr_desc);

    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::zero();
    auto condition2 = symbolic::Lt(indvar2, indvar1);
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Reduction block
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in1", {indvar2}, ptr_desc);
    builder.add_computational_memlet(block, A_in, tasklet, "_in2", {indvar1}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, ptr_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop1);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies1 = deps.dependencies(loop1);
    auto dependencies2 = deps.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_EQ(dependencies2.size(), 1);
    EXPECT_EQ(dependencies2.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(ScopAnalysisTest, DependenceInfoTest_Transpose_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("M"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);
    builder.add_container("B", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("M");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, A, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder
        .add_computational_memlet(block, tasklet, "_out", B, {symbolic::symbol("j"), symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop1);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies1 = deps.dependencies(loop1);
    auto dependencies2 = deps.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies2.size(), 0);
    EXPECT_EQ(dependencies1.size(), 0);
}

TEST(ScopAnalysisTest, DependenceInfoTest_TransposeTriangle_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::add(indvar1, symbolic::integer(1));
    auto condition2 = symbolic::Lt(indvar2, symbolic::symbol("N"));
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder
        .add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("j"), symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop1);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies1 = deps.dependencies(loop1);
    auto dependencies2 = deps.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 0);
    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(ScopAnalysisTest, DependenceInfoTest_TransposeTriangleWithDiagonal_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto indvar2 = symbolic::symbol("j");
    auto init2 = indvar1;
    auto condition2 = symbolic::Lt(indvar2, symbolic::symbol("N"));
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, edge_desc);
    builder
        .add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("j"), symbolic::symbol("i")}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop1);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies1 = deps.dependencies(loop1);
    auto dependencies2 = deps.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 0);
    EXPECT_EQ(dependencies2.size(), 0);
}

TEST(ScopAnalysisTest, DISABLED_DependenceInfoTest_TransposeSquare_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_desc(base_desc, symbolic::symbol("N"));
    types::Pointer edge_desc(array_desc);
    types::Pointer desc;

    builder.add_container("A", desc, true);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    auto bound2 = symbolic::symbol("N");
    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::integer(0);
    auto condition2 = symbolic::Lt(indvar2, bound2);
    auto update3 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update3);
    auto& body2 = loop2.root();

    // Add computation
    auto& block = builder.add_block(body2);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar1, indvar2}, edge_desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar2, indvar1}, edge_desc);

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop1);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies1 = deps.dependencies(loop1);
    auto dependencies2 = deps.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies2.size(), 0);
    EXPECT_EQ(dependencies1.size(), 2);
    EXPECT_EQ(dependencies1.at("A"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(ScopAnalysisTest, DependenceInfoTest_ReductionWithLocalStorage) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(base_desc);
    types::Array array_desc(base_desc, symbolic::integer(2));

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);
    builder.add_container("local", array_desc);

    // Define loop
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, symbolic::symbol("N"));
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_for(root, indvar1, condition1, init1, update1);
    auto& body1 = loop1.root();

    // local[0] = 0.0 block
    {
        auto& init_block = builder.add_block(body1);
        auto& local_init_0 = builder.add_access(init_block, "local");
        auto& zero_node = builder.add_constant(init_block, "0.0", base_desc);
        auto& tasklet_init = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(init_block, zero_node, tasklet_init, "_in", {});
        builder.add_computational_memlet(init_block, tasklet_init, "_out", local_init_0, {symbolic::zero()}, array_desc);
    }

    // local[1] = 0.0 block
    {
        auto& init_block = builder.add_block(body1);
        auto& local_init_0 = builder.add_access(init_block, "local");
        auto& zero_node = builder.add_constant(init_block, "0.0", base_desc);
        auto& tasklet_init = builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(init_block, zero_node, tasklet_init, "_in", {});
        builder.add_computational_memlet(init_block, tasklet_init, "_out", local_init_0, {symbolic::one()}, array_desc);
    }

    auto indvar2 = symbolic::symbol("j");
    auto init2 = symbolic::zero();
    auto condition2 = symbolic::Lt(indvar2, indvar1);
    auto update2 = symbolic::add(indvar2, symbolic::integer(1));

    auto& loop2 = builder.add_for(body1, indvar2, condition2, init2, update2);
    auto& body2 = loop2.root();

    // Reduction: local[0] += A[j] block
    {
        auto& block = builder.add_block(body2);
        auto& A_in = builder.add_access(block, "A");
        auto& local_in = builder.add_access(block, "local");
        auto& local_out = builder.add_access(block, "local");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, A_in, tasklet, "_in1", {indvar2}, ptr_desc);
        builder.add_computational_memlet(block, local_in, tasklet, "_in2", {symbolic::zero()}, array_desc);
        builder.add_computational_memlet(block, tasklet, "_out", local_out, {symbolic::zero()}, array_desc);
    }

    // Reduction: local[1] *= A[j] block
    {
        auto& block = builder.add_block(body2);
        auto& A_in = builder.add_access(block, "A");
        auto& local_in = builder.add_access(block, "local");
        auto& local_out = builder.add_access(block, "local");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, A_in, tasklet, "_in1", {indvar2}, ptr_desc);
        builder.add_computational_memlet(block, local_in, tasklet, "_in2", {symbolic::one()}, array_desc);
        builder.add_computational_memlet(block, tasklet, "_out", local_out, {symbolic::one()}, array_desc);
    }

    // Writeback block: B[i] = local[0]; C[i] = local[1]
    {
        auto& block = builder.add_block(body1);
        auto& local_in_0 = builder.add_access(block, "local");
        auto& local_in_1 = builder.add_access(block, "local");
        auto& B_out = builder.add_access(block, "B");
        auto& C_out = builder.add_access(block, "C");
        auto& tasklet_0 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, local_in_0, tasklet_0, "_in", {symbolic::zero()}, array_desc);
        builder.add_computational_memlet(block, tasklet_0, "_out", B_out, {indvar1}, ptr_desc);

        auto& tasklet_1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, local_in_1, tasklet_1, "_in", {symbolic::one()}, array_desc);
        builder.add_computational_memlet(block, tasklet_1, "_out", C_out, {indvar1}, ptr_desc);
    }

    // Analysis
    analysis::AnalysisManager am(sdfg);
    analysis::ScopBuilder scop_builder(sdfg, loop1);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);
    analysis::Dependences deps(*scop);
    auto dependencies1 = deps.dependencies(loop1);
    auto dependencies2 = deps.dependencies(loop2);

    // Check
    EXPECT_EQ(dependencies1.size(), 1);
    EXPECT_EQ(dependencies1.at("local"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
    EXPECT_EQ(dependencies2.size(), 1);
    EXPECT_EQ(dependencies2.at("local"), analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE);
}

TEST(ScopAnalysisTest, ScopToSDFGTest_SimpleLoopWithTasklet) {
    builder::StructuredSDFGBuilder builder("reconstruction_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer int_ptr_type(int_type);
    builder.add_container("i", int_type);
    builder.add_container("A", int_ptr_type);
    builder.add_container("B", int_ptr_type);

    auto i_sym = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        i_sym,
        symbolic::Lt(i_sym, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(i_sym, symbolic::integer(1))
    );

    // Add tasklet
    auto& block = builder.add_block(loop.root());
    auto& in_node = builder.add_access(block, "A");
    auto& out_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_int"});
    builder.add_computational_memlet(block, in_node, tasklet, "_int", {i_sym}, int_ptr_type);
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {i_sym}, int_ptr_type);

    // Run Analysis to build Scop
    analysis::AnalysisManager am(sdfg);
    am.get<analysis::ScopeAnalysis>();

    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);

    // Check initial statement count
    ASSERT_EQ(scop->statements().size(), 1);

    // 2. Convert Scop back to SDFG
    analysis::ScopToSDFG converter(*scop, builder);
    converter.build(am);

    ASSERT_EQ(root.size(), 1);
    auto* new_seq = dynamic_cast<structured_control_flow::Sequence*>(&root.at(0).first);
    ASSERT_NE(new_seq, nullptr);
    ASSERT_EQ(new_seq->size(), 1);

    auto* new_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&new_seq->at(0).first);
    ASSERT_NE(new_loop, nullptr);
    EXPECT_EQ(new_loop->indvar()->get_name(), i_sym->get_name());
    EXPECT_TRUE(symbolic::eq(new_loop->condition(), symbolic::Le(i_sym, symbolic::integer(9))));
    EXPECT_TRUE(symbolic::eq(new_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(new_loop->update(), symbolic::add(i_sym, symbolic::integer(1))));

    auto& inner_seq = new_loop->root();
    ASSERT_EQ(inner_seq.size(), 1);

    auto inner_block = dynamic_cast<structured_control_flow::Block*>(&inner_seq.at(0).first);
    ASSERT_NE(inner_block, nullptr);
    auto& dfg = inner_block->dataflow();
    ASSERT_EQ(dfg.nodes().size(), 3); // A access, tasklet, B access
    ASSERT_EQ(dfg.edges().size(), 2); // A -> tasklet, tasklet -> B

    bool found_tasklet = false;
    bool found_A_access = false;
    bool found_B_access = false;
    for (const auto& node : dfg.nodes()) {
        if (auto* tasklet_node = dynamic_cast<const data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::assign);
            ASSERT_EQ(found_tasklet, false); // Ensure only one tasklet
            found_tasklet = true;
        } else if (auto* access_node = dynamic_cast<const data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                ASSERT_EQ(found_A_access, false); // Ensure only one A access
                found_A_access = true;
            } else if (access_node->data() == "B") {
                ASSERT_EQ(found_B_access, false); // Ensure only one B access
                found_B_access = true;
            }
        }
    }
    ASSERT_TRUE(found_tasklet);
    ASSERT_TRUE(found_A_access);
    ASSERT_TRUE(found_B_access);
}

TEST(ScopAnalysisTest, ScopToSDFGTest_SimpleLoopWithExpression) {
    builder::StructuredSDFGBuilder builder("reconstruction_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("A", int_type);
    builder.add_container("B", int_type);

    auto i_sym = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        i_sym,
        symbolic::Lt(i_sym, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(i_sym, symbolic::integer(1))
    );

    // Add tasklet
    auto& block = builder.add_block(loop.root(), {{symbolic::symbol("A"), symbolic::symbol("B")}});

    // Run Analysis to build Scop
    analysis::AnalysisManager am(sdfg);
    am.get<analysis::ScopeAnalysis>();

    analysis::ScopBuilder scop_builder(sdfg, loop);
    auto scop = scop_builder.build(am);
    ASSERT_NE(scop, nullptr);

    // Check initial statement count
    ASSERT_EQ(scop->statements().size(), 1);

    // 2. Convert Scop back to SDFG
    analysis::ScopToSDFG converter(*scop, builder);
    converter.build(am);

    ASSERT_EQ(root.size(), 1);
    auto* new_seq = dynamic_cast<structured_control_flow::Sequence*>(&root.at(0).first);
    ASSERT_NE(new_seq, nullptr);
    ASSERT_EQ(new_seq->size(), 1);

    auto* new_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&new_seq->at(0).first);
    ASSERT_NE(new_loop, nullptr);
    EXPECT_EQ(new_loop->indvar()->get_name(), i_sym->get_name());
    EXPECT_TRUE(symbolic::eq(new_loop->condition(), symbolic::Le(i_sym, symbolic::integer(9))));
    EXPECT_TRUE(symbolic::eq(new_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(new_loop->update(), symbolic::add(i_sym, symbolic::integer(1))));

    auto& inner_seq = new_loop->root();
    ASSERT_EQ(inner_seq.size(), 1);

    auto inner_block = dynamic_cast<structured_control_flow::Block*>(&inner_seq.at(0).first);
    ASSERT_NE(inner_block, nullptr);
    auto& dfg = inner_block->dataflow();
    ASSERT_EQ(dfg.nodes().size(), 0);
    ASSERT_EQ(dfg.edges().size(), 0);

    EXPECT_EQ(inner_seq.at(0).second.size(), 1);
    EXPECT_TRUE(symbolic::eq((*inner_seq.at(0).second.assignments().begin()).first, symbolic::symbol("A")));
    EXPECT_TRUE(symbolic::eq((*inner_seq.at(0).second.assignments().begin()).second, symbolic::symbol("B")));
}
