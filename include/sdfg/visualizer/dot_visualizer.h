#pragma once

#include <string>

#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/visualizer/visualizer.h"

namespace sdfg {
namespace visualizer {

class DotVisualizer : public Visualizer {
   private:
    std::string last_comp_name_;
    std::string last_comp_name_cluster_;

    virtual void visualizeBlock(StructuredSDFG& sdfg,
                                structured_control_flow::Block& block) override;
    virtual void visualizeSequence(StructuredSDFG& sdfg,
                                   structured_control_flow::Sequence& sequence) override;
    virtual void visualizeIfElse(StructuredSDFG& sdfg,
                                 structured_control_flow::IfElse& if_else) override;
    virtual void visualizeWhile(StructuredSDFG& sdfg,
                                structured_control_flow::While& while_loop) override;
    virtual void visualizeFor(StructuredSDFG& sdfg, structured_control_flow::For& loop) override;
    virtual void visualizeReturn(StructuredSDFG& sdfg,
                                 structured_control_flow::Return& return_node) override;
    virtual void visualizeBreak(StructuredSDFG& sdfg,
                                structured_control_flow::Break& break_node) override;
    virtual void visualizeContinue(StructuredSDFG& sdfg,
                                   structured_control_flow::Continue& continue_node) override;
    virtual void visualizeMap(StructuredSDFG& sdfg,
                              structured_control_flow::Map& map_node) override;

   public:
    using Visualizer::Visualizer;

    virtual void visualize() override;
};

}  // namespace visualizer
}  // namespace sdfg
