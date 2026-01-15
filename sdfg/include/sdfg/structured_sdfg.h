#pragma once

#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <ranges>
#include <string>
#include <unordered_map>
#include <unordered_set>

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
    std::unique_ptr<structured_control_flow::Sequence> root_;

public:
    StructuredSDFG(const std::string& name, FunctionType type);
    StructuredSDFG(const std::string& name, FunctionType type, const types::IType& return_type);

    StructuredSDFG(const StructuredSDFG& sdfg) = delete;
    StructuredSDFG& operator=(const StructuredSDFG&) = delete;

    void validate() const override;

    const structured_control_flow::Sequence& root() const;

    structured_control_flow::Sequence& root();

    std::unique_ptr<StructuredSDFG> clone() const;
};

} // namespace sdfg
