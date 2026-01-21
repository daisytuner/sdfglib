#include "sdfg/passes/pipeline.h"

#include "sdfg/passes/offloading/code_motion/block_hoisting.h"
#include "sdfg/passes/offloading/code_motion/block_sorting.h"

namespace sdfg {
namespace passes {

Pipeline code_motion() {
    Pipeline p("CodeMotion");

    p.register_pass<BlockHoistingPass>();
    p.register_pass<BlockSortingPass>();

    return p;
};

} // namespace passes
} // namespace sdfg
