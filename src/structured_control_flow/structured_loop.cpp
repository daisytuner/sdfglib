#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace structured_control_flow {

StructuredLoop::StructuredLoop(
    size_t element_id,
    const DebugInfo& debug_info,
    symbolic::Symbol indvar,
    symbolic::Expression init,
    symbolic::Expression update,
    symbolic::Condition condition
)
    : ControlFlowNode(element_id, debug_info), indvar_(indvar), init_(init), update_(update), condition_(condition) {
    this->root_ = std::unique_ptr<Sequence>(new Sequence(++element_id, debug_info));
}

void StructuredLoop::validate(const Function& function) const {
    if (this->indvar_.is_null()) {
        throw InvalidSDFGException("StructuredLoop: Induction variable cannot be null");
    }
    if (this->init_.is_null()) {
        throw InvalidSDFGException("StructuredLoop: Initialization expression cannot be null");
    }
    if (this->update_.is_null()) {
        throw InvalidSDFGException("StructuredLoop: Update expression cannot be null");
    }
    if (this->condition_.is_null()) {
        throw InvalidSDFGException("StructuredLoop: Condition expression cannot be null");
    }
    if (!SymEngine::is_a_Boolean(*this->condition_)) {
        throw InvalidSDFGException("StructuredLoop: Condition expression must be a boolean expression");
    }

    this->root_->validate(function);
};

const symbolic::Symbol StructuredLoop::indvar() const { return this->indvar_; };

const symbolic::Expression StructuredLoop::init() const { return this->init_; };

const symbolic::Expression StructuredLoop::update() const { return this->update_; };

const symbolic::Condition StructuredLoop::condition() const { return this->condition_; };

Sequence& StructuredLoop::root() const { return *this->root_; };

void StructuredLoop::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    if (symbolic::eq(this->indvar_, old_expression) && SymEngine::is_a<SymEngine::Symbol>(*new_expression)) {
        this->indvar_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    this->init_ = symbolic::subs(this->init_, old_expression, new_expression);
    this->update_ = symbolic::subs(this->update_, old_expression, new_expression);
    this->condition_ = symbolic::subs(this->condition_, old_expression, new_expression);

    this->root_->replace(old_expression, new_expression);
};

} // namespace structured_control_flow
} // namespace sdfg
