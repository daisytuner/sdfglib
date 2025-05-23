#pragma once

#include <string>
#include <utility>
#include <vector>

#include "sdfg/codegen/utils.h"
#include "sdfg/conditional_schedule.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/schedule.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/kernel.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace visualizer {

class Visualizer {
   protected:
    codegen::PrettyPrinter stream_;
    ConditionalSchedule& schedule_;
    std::vector<std::pair<const std::string, const std::string>> replacements_;

    virtual std::string expression(const std::string expr);

    virtual void visualizeNode(Schedule& schedule, structured_control_flow::ControlFlowNode& node);
    virtual void visualizeBlock(Schedule& schedule, structured_control_flow::Block& block) = 0;
    virtual void visualizeSequence(Schedule& schedule,
                                   structured_control_flow::Sequence& sequence) = 0;
    virtual void visualizeIfElse(Schedule& schedule, structured_control_flow::IfElse& if_else) = 0;
    virtual void visualizeWhile(Schedule& schedule, structured_control_flow::While& while_loop) = 0;
    virtual void visualizeFor(Schedule& schedule, structured_control_flow::For& loop) = 0;
    virtual void visualizeReturn(Schedule& schedule,
                                 structured_control_flow::Return& return_node) = 0;
    virtual void visualizeBreak(Schedule& schedule, structured_control_flow::Break& break_node) = 0;
    virtual void visualizeContinue(Schedule& schedule,
                                   structured_control_flow::Continue& continue_node) = 0;
    virtual void visualizeKernel(Schedule& schedule,
                                 structured_control_flow::Kernel& kernel_node) = 0;
    virtual void visualizeMap(Schedule& schedule, structured_control_flow::Map& map_node) = 0;

    virtual void visualizeTasklet(data_flow::Tasklet const& tasklet);
    virtual void visualizeForBounds(symbolic::Symbol const& indvar,
                                    symbolic::Expression const& init,
                                    symbolic::Condition const& condition,
                                    symbolic::Expression const& update);
    virtual void visualizeLibraryNode(const data_flow::LibraryNodeType libnode_type);
    virtual void visualizeSubset(Function const& function, types::IType const& type,
                                 data_flow::Subset const& sub);

   public:
    Visualizer(ConditionalSchedule& schedule) : stream_{}, schedule_{schedule}, replacements_{} {};

    virtual void visualize() = 0;

    codegen::PrettyPrinter const& getStream() const { return this->stream_; }
};

}  // namespace visualizer
}  // namespace sdfg
