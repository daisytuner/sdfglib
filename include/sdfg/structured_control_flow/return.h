#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class Return : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    std::string data_;

    Return(size_t element_id, const DebugInfo& debug_info, const std::string& data);

public:
    Return(const Return& Return) = delete;
    Return& operator=(const Return&) = delete;

    bool has_data() const;

    const std::string& data() const;

    void validate(const Function& function) const override;

    void replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) override;
};

} // namespace structured_control_flow
} // namespace sdfg
