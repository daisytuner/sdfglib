#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <string>

#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"

namespace sdfg {

class Schedule;
namespace builder {
class StructuredSDFGBuilder;
}

class StructuredSDFG : public Function {
    friend class sdfg::builder::StructuredSDFGBuilder;
    friend class Schedule;

private:
    std::unique_ptr<structured_control_flow::Sequence> root;

public:
    StructuredSDFG(const std::string& name, FunctionType type);

    StructuredSDFG(const StructuredSDFG& sdfg) = delete;
    auto operator=(const StructuredSDFG&) -> StructuredSDFG& = delete;

    auto debug_info() const -> const DebugInfo override;

    auto root() const -> const structured_control_flow::Sequence&;

    auto root() -> structured_control_flow::Sequence&;

    auto clone() const -> std::unique_ptr<StructuredSDFG>;

    auto num_nodes() const -> size_t;
};

} // namespace sdfg
