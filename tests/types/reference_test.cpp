#include <gtest/gtest.h>

#include "sdfg/types/type.h"
#include "sdfg/types/scalar.h"
#include "sdfg/codegen/utils.h"

using namespace sdfg;

TEST(ReferenceTests, TypeId) {
    codegen::Reference r(types::StorageType_CPU_Stack, 4, "ref_init", types::Scalar(types::PrimitiveType::Int32));
    
    EXPECT_EQ(r.type_id(), types::TypeID::Reference);
}
