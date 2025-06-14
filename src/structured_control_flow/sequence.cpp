#include "sdfg/structured_control_flow/sequence.h"

namespace sdfg {
namespace structured_control_flow {

Transition::Transition(const DebugInfo& debug_info, Sequence& parent)
    : Element(debug_info), parent_(parent) {

      };

Transition::Transition(const DebugInfo& debug_info, Sequence& parent,
                       const control_flow::Assignments& assignments)
    : Element(debug_info), parent_(parent), assignments_(assignments) {

      };

const control_flow::Assignments& Transition::assignments() const { return this->assignments_; };

control_flow::Assignments& Transition::assignments() { return this->assignments_; };

Sequence& Transition::parent() { return this->parent_; };

const Sequence& Transition::parent() const { return this->parent_; };

bool Transition::empty() const { return this->assignments_.empty(); };

size_t Transition::size() const { return this->assignments_.size(); };

void Transition::replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) {

};

Sequence::Sequence(const DebugInfo& debug_info)
    : ControlFlowNode(debug_info) {

      };

size_t Sequence::size() const { return this->children_.size(); };

std::pair<const ControlFlowNode&, const Transition&> Sequence::at(size_t i) const {
    return {*this->children_.at(i), *this->transitions_.at(i)};
};

std::pair<ControlFlowNode&, Transition&> Sequence::at(size_t i) {
    return {*this->children_.at(i), *this->transitions_.at(i)};
};

void Sequence::replace(const symbolic::Expression& old_expression,
                       const symbolic::Expression& new_expression) {
    for (auto& child : this->children_) {
        child->replace(old_expression, new_expression);
    }

    if (SymEngine::is_a<SymEngine::Symbol>(*old_expression)) {
        auto old_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(old_expression);

        for (auto& trans : this->transitions_) {
            if (trans->assignments().find(old_symbol) != trans->assignments().end()) {
                if (!SymEngine::is_a<SymEngine::Symbol>(*new_expression)) {
                    throw InvalidSDFGException(
                        "Sequence: Assigments do not support complex expressions on LHS");
                }
                auto new_symbol =
                    SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
                trans->assignments()[new_symbol] = trans->assignments()[old_symbol];
                trans->assignments().erase(old_symbol);
            }

            for (auto& entry : trans->assignments()) {
                entry.second = symbolic::subs(entry.second, old_expression, new_expression);
            }
        }
    }
};

}  // namespace structured_control_flow
}  // namespace sdfg
