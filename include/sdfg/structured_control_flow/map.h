#pragma once

#include <memory>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class Map : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    symbolic::Symbol indvar_;
    symbolic::Expression num_iterations_;

    std::unique_ptr<Sequence> root_;

    Map(size_t element_id, const DebugInfo& debug_info, symbolic::Symbol indvar,
        symbolic::Expression num_iterations);

   public:
    Map(const Map& node) = delete;
    Map& operator=(const Map&) = delete;

    const symbolic::Symbol& indvar() const;

    symbolic::Symbol& indvar();

    const symbolic::Expression& num_iterations() const;

    symbolic::Expression& num_iterations();

    Sequence& root() const;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace structured_control_flow
}  // namespace sdfg