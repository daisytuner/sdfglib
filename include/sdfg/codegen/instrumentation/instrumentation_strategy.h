#pragma once

#include "sdfg/codegen/instrumentation/instrumentation.h"
#include "sdfg/codegen/instrumentation/outermost_loops_instrumentation.h"

namespace sdfg {
namespace codegen {

enum InstrumentationStrategy { NONE, OUTERMOST_LOOPS };

inline std::unique_ptr<Instrumentation> create_instrumentation(InstrumentationStrategy strategy,
                                                               StructuredSDFG& sdfg) {
    switch (strategy) {
        case InstrumentationStrategy::NONE:
            return std::make_unique<Instrumentation>(sdfg);
        case InstrumentationStrategy::OUTERMOST_LOOPS:
            return std::make_unique<OutermostLoopsInstrumentation>(sdfg);
    }
}

}  // namespace codegen
}  // namespace sdfg
