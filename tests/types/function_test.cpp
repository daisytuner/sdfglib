#include "sdfg/types/function.h"

#include <gtest/gtest.h>

#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(FunctionTest, ReturnType) {
    types::Function f(types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(f.return_type(), types::Scalar(types::PrimitiveType::Int32));
}

TEST(FunctionTest, Params) {
    types::Function f(types::Scalar(types::PrimitiveType::Int32));
    f.add_param(types::Scalar(types::PrimitiveType::Int64));
    f.add_param(types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(f.num_params(), 2);
    EXPECT_EQ(f.param_type(symbolic::integer(0)), types::Scalar(types::PrimitiveType::Int64));
    EXPECT_EQ(f.param_type(symbolic::integer(1)), types::Scalar(types::PrimitiveType::Int32));
}

TEST(FunctionTest, IsVarArg) {
    types::Function f(types::Scalar(types::PrimitiveType::Int32));
    EXPECT_FALSE(f.is_var_arg());

    types::Function f2(types::Scalar(types::PrimitiveType::Int32), true);
    EXPECT_TRUE(f2.is_var_arg());
}