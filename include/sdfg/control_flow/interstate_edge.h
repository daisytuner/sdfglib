#pragma once

#include <boost/lexical_cast.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

#include "sdfg/control_flow/state.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

using json = nlohmann::json;

namespace sdfg {

namespace builder {
class SDFGBuilder;
}

namespace control_flow {

class InterstateEdge : public Element {
    friend class builder::SDFGBuilder;

   private:
    // Remark: Exclusive resource
    const graph::Edge edge_;

    const control_flow::State& src_;
    const control_flow::State& dst_;

    symbolic::Condition condition_;
    symbolic::Assignments assignments_;

    InterstateEdge(size_t element_id, const DebugInfo& debug_info, const graph::Edge& edge,
                   const control_flow::State& src, const control_flow::State& dst,
                   const symbolic::Condition& condition, const symbolic::Assignments& assignments);

   public:
    // Remark: Exclusive resource
    InterstateEdge(const InterstateEdge& state) = delete;
    InterstateEdge& operator=(const InterstateEdge&) = delete;

    const graph::Edge edge() const;

    const control_flow::State& src() const;

    const control_flow::State& dst() const;

    const symbolic::Condition& condition() const;

    bool is_unconditional() const;

    const symbolic::Assignments& assignments() const;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace control_flow
}  // namespace sdfg
