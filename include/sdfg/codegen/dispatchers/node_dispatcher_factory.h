#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/dispatchers/for_dispatcher.h"
#include "sdfg/codegen/dispatchers/if_else_dispatcher.h"
#include "sdfg/codegen/dispatchers/kernel_dispatcher.h"
#include "sdfg/codegen/dispatchers/map_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"
#include "sdfg/codegen/dispatchers/while_dispatcher.h"
#include "sdfg/codegen/language_extension.h"

namespace sdfg {
namespace codegen {

inline std::unique_ptr<NodeDispatcher> create_dispatcher(
    LanguageExtension& language_extension, Schedule& schedule,
    structured_control_flow::ControlFlowNode& node, Instrumentation& instrumentation) {
    if (auto block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        return std::make_unique<BlockDispatcher>(language_extension, schedule, *block,
                                                 instrumentation);
    } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        return std::make_unique<SequenceDispatcher>(language_extension, schedule, *sequence,
                                                    instrumentation);
    } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        return std::make_unique<IfElseDispatcher>(language_extension, schedule, *if_else,
                                                  instrumentation);
    } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        return std::make_unique<WhileDispatcher>(language_extension, schedule, *while_loop,
                                                 instrumentation);
    } else if (auto loop = dynamic_cast<structured_control_flow::For*>(&node)) {
        return std::make_unique<ForDispatcher>(language_extension, schedule, *loop,
                                               instrumentation);
    } else if (auto return_node = dynamic_cast<structured_control_flow::Return*>(&node)) {
        return std::make_unique<ReturnDispatcher>(language_extension, schedule, *return_node,
                                                  instrumentation);
    } else if (auto break_node = dynamic_cast<structured_control_flow::Break*>(&node)) {
        return std::make_unique<BreakDispatcher>(language_extension, schedule, *break_node,
                                                 instrumentation);
    } else if (auto continue_node = dynamic_cast<structured_control_flow::Continue*>(&node)) {
        return std::make_unique<ContinueDispatcher>(language_extension, schedule, *continue_node,
                                                    instrumentation);
    } else if (auto kernel_node = dynamic_cast<structured_control_flow::Kernel*>(&node)) {
        return std::make_unique<KernelDispatcher>(language_extension, schedule, *kernel_node,
                                                  instrumentation);
    } else if (auto map_node = dynamic_cast<structured_control_flow::Map*>(&node)) {
        return std::make_unique<MapDispatcher>(language_extension, schedule, *map_node,
                                               instrumentation);
    } else {
        throw std::runtime_error("Unsupported control flow node");
    }
};

}  // namespace codegen
}  // namespace sdfg
