#pragma once

#include <string>
#include <utility>
#include <vector>

#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace visualizer {

class Visualizer {
   protected:
    codegen::PrettyPrinter stream_;
    StructuredSDFG& sdfg_;
    std::vector<std::pair<const std::string, const std::string>> replacements_;

    virtual std::string expression(const std::string expr);

    virtual void visualizeNode(StructuredSDFG& sdfg,
                               structured_control_flow::ControlFlowNode& node);
    virtual void visualizeBlock(StructuredSDFG& sdfg, structured_control_flow::Block& block) = 0;
    virtual void visualizeSequence(StructuredSDFG& sdfg,
                                   structured_control_flow::Sequence& sequence) = 0;
    virtual void visualizeIfElse(StructuredSDFG& sdfg,
                                 structured_control_flow::IfElse& if_else) = 0;
    virtual void visualizeWhile(StructuredSDFG& sdfg,
                                structured_control_flow::While& while_loop) = 0;
    virtual void visualizeFor(StructuredSDFG& sdfg, structured_control_flow::For& loop) = 0;
    virtual void visualizeReturn(StructuredSDFG& sdfg,
                                 structured_control_flow::Return& return_node) = 0;
    virtual void visualizeBreak(StructuredSDFG& sdfg,
                                structured_control_flow::Break& break_node) = 0;
    virtual void visualizeContinue(StructuredSDFG& sdfg,
                                   structured_control_flow::Continue& continue_node) = 0;
    virtual void visualizeMap(StructuredSDFG& sdfg, structured_control_flow::Map& map_node) = 0;

    virtual void visualizeTasklet(data_flow::Tasklet const& tasklet);
    virtual void visualizeForBounds(symbolic::Symbol const& indvar,
                                    symbolic::Expression const& init,
                                    symbolic::Condition const& condition,
                                    symbolic::Expression const& update);
    virtual void visualizeSubset(Function const& function, types::IType const& type,
                                 data_flow::Subset const& sub);

   public:
    Visualizer(StructuredSDFG& sdfg) : stream_{}, sdfg_{sdfg}, replacements_{} {};

    virtual void visualize() = 0;

    codegen::PrettyPrinter const& getStream() const { return this->stream_; }
};

}  // namespace visualizer
}  // namespace sdfg
