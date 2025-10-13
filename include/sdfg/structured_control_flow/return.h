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
    std::unique_ptr<types::IType> type_;
    bool unreachable_;

    Return(size_t element_id, const DebugInfo& debug_info, const std::string& data);

    Return(size_t element_id, const DebugInfo& debug_info);

    Return(size_t element_id, const DebugInfo& debug_info, const std::string& constant, const types::IType& type);

public:
    Return(const Return& Return) = delete;
    Return& operator=(const Return&) = delete;

    const std::string& data() const;

    const types::IType& type() const;

    bool unreachable() const;

    bool is_data() const;

    bool is_unreachable() const;

    bool is_constant() const;

    void validate(const Function& function) const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

} // namespace structured_control_flow
} // namespace sdfg
