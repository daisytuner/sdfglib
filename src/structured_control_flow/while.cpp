#include "sdfg/structured_control_flow/while.h"

namespace sdfg {
namespace structured_control_flow {

While::While(const DebugInfo& debug_info) : ControlFlowNode(debug_info) {
    this->root_ = std::unique_ptr<Sequence>(new Sequence(debug_info));
};

const Sequence& While::root() const { return *this->root_; };

Sequence& While::root() { return *this->root_; };

void While::replace(const symbolic::Expression& old_expression,
                    const symbolic::Expression& new_expression) {
    this->root_->replace(old_expression, new_expression);
};

Break::Break(const DebugInfo& debug_info)
    : ControlFlowNode(debug_info) {

      };

void Break::replace(const symbolic::Expression& old_expression,
                    const symbolic::Expression& new_expression) {

};

Continue::Continue(const DebugInfo& debug_info)
    : ControlFlowNode(debug_info) {

      };

void Continue::replace(const symbolic::Expression& old_expression,
                       const symbolic::Expression& new_expression) {

};

}  // namespace structured_control_flow
}  // namespace sdfg
