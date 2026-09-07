#pragma once

#include "sdfg/einsum/einsum_node.h"
#include "sdfg/einsum/passes/einsum_passes.h"
#include "sdfg/einsum/transformations/einsum2dot.h"
#include "sdfg/einsum/transformations/einsum2gemm.h"
#include "sdfg/einsum/transformations/einsum_extend.h"
#include "sdfg/einsum/transformations/einsum_lift.h"
#include "sdfg/einsum/transformations/einsum_promotion.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace einsum {

inline void register_einsum_plugin() {
    serializer::LibraryNodeSerializerRegistry::instance()
        .register_library_node_serializer(LibraryNodeType_Einsum.value(), []() {
            return std::make_unique<EinsumSerializer>();
        });
}

} // namespace einsum
} // namespace sdfg
