#pragma once

#include <string>

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/schedule.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/kernel.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/visualizer/visualizer.h"

namespace sdfg {
namespace visualizer {

class DotVisualizer : public Visualizer {
   private:
    std::string last_comp_name_;
    std::string last_comp_name_cluster_;

    codegen::CLanguageExtension le_;

    virtual void visualizeBlock(Schedule& schedule, structured_control_flow::Block& block) override;
    virtual void visualizeSequence(Schedule& schedule,
                                   structured_control_flow::Sequence& sequence) override;
    virtual void visualizeIfElse(Schedule& schedule,
                                 structured_control_flow::IfElse& if_else) override;
    virtual void visualizeWhile(Schedule& schedule,
                                structured_control_flow::While& while_loop) override;
    virtual void visualizeFor(Schedule& schedule, structured_control_flow::For& loop) override;
    virtual void visualizeReturn(Schedule& schedule,
                                 structured_control_flow::Return& return_node) override;
    virtual void visualizeBreak(Schedule& schedule,
                                structured_control_flow::Break& break_node) override;
    virtual void visualizeContinue(Schedule& schedule,
                                   structured_control_flow::Continue& continue_node) override;
    virtual void visualizeKernel(Schedule& schedule,
                                 structured_control_flow::Kernel& kernel_node) override;

   public:
    using Visualizer::Visualizer;

    virtual void visualize() override;
};

}  // namespace visualizer
}  // namespace sdfg
