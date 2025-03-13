#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
using namespace sdfg;

TEST(StructuredSDFGTest, Clone) {
    builder::StructuredSDFGBuilder builder("sdfg_1");

    types::Scalar desc(types::PrimitiveType::UInt64);
    builder.add_container("N", desc, true);
    builder.add_container("M", desc, false, true);
    builder.add_container("i", desc);

    builder.add_structure("struct_1");
    types::Structure desc_struct("struct_1");
    builder.add_container("S", desc_struct);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(
        root, symbolic::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});

    auto sdfg = builder.move();

    auto cloned_sdfg = sdfg->clone();
    EXPECT_EQ(cloned_sdfg->name(), "sdfg_1");
    EXPECT_EQ(cloned_sdfg->containers().size(), 4);
    EXPECT_EQ(cloned_sdfg->arguments().size(), 1);
    EXPECT_EQ(cloned_sdfg->externals().size(), 1);
    EXPECT_EQ(cloned_sdfg->structures().size(), 1);

    auto& cloned_root = cloned_sdfg->root();
    EXPECT_EQ(cloned_root.size(), 1);

    auto cloned_block = cloned_root.at(0);
    EXPECT_TRUE(dynamic_cast<const structured_control_flow::Block*>(&cloned_block.first));
    EXPECT_EQ(cloned_block.second.size(), 1);
    EXPECT_TRUE(symbolic::eq(cloned_block.second.assignments().at(symbolic::symbol("N")),
                             symbolic::integer(10)));
}
