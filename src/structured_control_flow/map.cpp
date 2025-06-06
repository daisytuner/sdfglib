#include "sdfg/structured_control_flow/map.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace structured_control_flow {

Map::Map(const DebugInfo& debug_info, symbolic::Symbol indvar, symbolic::Expression num_iterations)
    : ControlFlowNode(debug_info), indvar_(indvar), num_iterations_(num_iterations) {
    this->root_ = std::unique_ptr<Sequence>(new Sequence(debug_info));
};

const symbolic::Symbol& Map::indvar() const { return this->indvar_; };

symbolic::Symbol& Map::indvar() { return this->indvar_; };

const symbolic::Expression& Map::num_iterations() const { return this->num_iterations_; };

symbolic::Expression& Map::num_iterations() { return this->num_iterations_; };

Sequence& Map::root() const { return *this->root_; };

void Map::replace(const symbolic::Expression& old_expression,
                  const symbolic::Expression& new_expression) {
    if (symbolic::eq(this->indvar_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression) &&
               "New Indvar must be a symbol");
        this->indvar_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    this->num_iterations_ = symbolic::subs(this->num_iterations_, old_expression, new_expression);

    this->root_->replace(old_expression, new_expression);
};

}  // namespace structured_control_flow
}  // namespace sdfg