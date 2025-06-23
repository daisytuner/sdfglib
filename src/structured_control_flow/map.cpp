#include "sdfg/structured_control_flow/map.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace structured_control_flow {

Map::Map(size_t element_id, const DebugInfo& debug_info, symbolic::Symbol indvar,
         symbolic::Expression init, symbolic::Expression update, symbolic::Condition condition,
         const ScheduleType& schedule_type)
    : StructuredLoop(element_id, debug_info),
      indvar_(indvar),
      schedule_type_(schedule_type),
      init_(init),
      update_(update),
      condition_(condition) {
    this->root_ = std::unique_ptr<Sequence>(new Sequence(++element_id, debug_info));
};

const symbolic::Symbol& Map::indvar() const { return this->indvar_; };

symbolic::Symbol& Map::indvar() { return this->indvar_; };

const symbolic::Expression& Map::init() const { return this->init_; };

const symbolic::Expression& Map::update() const { return this->update_; };

const symbolic::Condition& Map::condition() const { return this->condition_; };

ScheduleType& Map::schedule_type() { return this->schedule_type_; };

const ScheduleType& Map::schedule_type() const { return this->schedule_type_; };

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