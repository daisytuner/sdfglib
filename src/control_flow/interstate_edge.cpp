#include "sdfg/control_flow/interstate_edge.h"

namespace sdfg {
namespace control_flow {

InterstateEdge::InterstateEdge(size_t element_id, const DebugInfo& debug_info,
                               const graph::Edge& edge, const control_flow::State& src,
                               const control_flow::State& dst, const symbolic::Condition& condition,
                               const sdfg::symbolic::Assignments& assignments)
    : Element(element_id, debug_info),
      edge_(edge),
      src_(src),
      dst_(dst),
      condition_(condition),
      assignments_(assignments){

      };

const graph::Edge InterstateEdge::edge() const { return this->edge_; };

const control_flow::State& InterstateEdge::src() const { return this->src_; };

const control_flow::State& InterstateEdge::dst() const { return this->dst_; };

const symbolic::Condition& InterstateEdge::condition() const { return this->condition_; };

bool InterstateEdge::is_unconditional() const { return symbolic::is_true(this->condition_); };

const sdfg::symbolic::Assignments& InterstateEdge::assignments() const {
    return this->assignments_;
};

void InterstateEdge::replace(const symbolic::Expression& old_expression,
                             const symbolic::Expression& new_expression) {
    symbolic::subs(this->condition_, old_expression, new_expression);

    if (SymEngine::is_a<SymEngine::Symbol>(*old_expression)) {
        auto old_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(old_expression);
        if (this->assignments_.find(old_symbol) != this->assignments_.end()) {
            assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression) &&
                   "Assigments do not support complex expressions on LHS");
            auto new_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
            this->assignments_[new_symbol] = this->assignments_[old_symbol];
            this->assignments_.erase(old_symbol);
        }
    }

    for (auto& entry : this->assignments_) {
        entry.second = symbolic::subs(entry.second, old_expression, new_expression);
    }
};

}  // namespace control_flow
}  // namespace sdfg
