#pragma once

#include <memory>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class For : public StructuredLoop {
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    For(size_t element_id, const DebugInfo& debug_info, symbolic::Symbol indvar,
        symbolic::Expression init, symbolic::Expression update, symbolic::Condition condition);

   public:
    For(const For& node) = delete;
    For& operator=(const For&) = delete;
};

}  // namespace structured_control_flow
}  // namespace sdfg