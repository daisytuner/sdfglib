#pragma once

#include <sdfg/passes/pipeline.h>

#include "sdfg/passes/normalization/perfect_loop_distribution.h"
#include "sdfg/passes/normalization/stride_minimization.h"

namespace sdfg {
namespace passes {
namespace normalization {

inline passes::Pipeline loop_normalization() {
    passes::Pipeline pipeline("Loop Normalization");

    // Register passes for loop normalization
    pipeline.register_pass<normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<normalization::StrideMinimization>();

    return pipeline;
}

} // namespace normalization
} // namespace passes
} // namespace sdfg
