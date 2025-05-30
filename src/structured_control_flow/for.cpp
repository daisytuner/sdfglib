#include "sdfg/structured_control_flow/for.h"

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace structured_control_flow {

For::For(size_t element_id, const DebugInfo& debug_info, symbolic::Symbol indvar,
         symbolic::Expression init, symbolic::Expression update, symbolic::Condition condition)
    : StructuredLoop(element_id, debug_info),
      indvar_(indvar),
      init_(init),
      update_(update),
      condition_(condition) {
    this->root_ = std::unique_ptr<Sequence>(new Sequence(++element_id, debug_info));
};

const symbolic::Symbol& For::indvar() const { return this->indvar_; };

symbolic::Symbol& For::indvar() { return this->indvar_; };

const symbolic::Expression& For::init() const { return this->init_; };

symbolic::Expression& For::init() { return this->init_; };

const symbolic::Expression& For::update() const { return this->update_; };

symbolic::Expression& For::update() { return this->update_; };

const symbolic::Condition& For::condition() const { return this->condition_; };

symbolic::Condition& For::condition() { return this->condition_; };

Sequence& For::root() const { return *this->root_; };

void For::replace(const symbolic::Expression& old_expression,
                  const symbolic::Expression& new_expression) {
    if (symbolic::eq(this->indvar_, old_expression)) {
        if (!SymEngine::is_a<SymEngine::Symbol>(*new_expression)) {
            throw InvalidSDFGException("For: New Indvar must be a symbol");
        }
        this->indvar_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    this->init_ = symbolic::subs(this->init_, old_expression, new_expression);
    this->update_ = symbolic::subs(this->update_, old_expression, new_expression);
    this->condition_ = symbolic::subs(this->condition_, old_expression, new_expression);

    this->root_->replace(old_expression, new_expression);
};

}  // namespace structured_control_flow
}  // namespace sdfg