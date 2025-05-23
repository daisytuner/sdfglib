#include "sdfg/passes/structured_control_flow/sequence_fusion.h"

#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace passes {

SequenceFusion::SequenceFusion()
    : Pass() {

      };

std::string SequenceFusion::name() { return "SequenceFusion"; };

bool SequenceFusion::run_pass(builder::StructuredSDFGBuilder& builder,
                              analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    // Traverse structured SDFG
    std::list<structured_control_flow::ControlFlowNode*> queue = {&builder.subject().root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // If sequence, attempt fusion
        if (auto seq = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            size_t i = 0;
            while (i < seq->size()) {
                auto child = seq->at(i);
                auto subseq = dynamic_cast<structured_control_flow::Sequence*>(&child.first);
                if (!subseq) {
                    i++;
                    continue;
                }
                builder.insert_children(*seq, *subseq, i + 1);
                builder.remove_child(*seq, i);
                applied = true;
            }
        }

        // Add children to queue
        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            queue.push_back(&loop_stmt->root());
        } else if (auto sloop_stmt =
                       dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&sloop_stmt->root());
        } else if (auto kern_stmt = dynamic_cast<const structured_control_flow::Kernel*>(current)) {
            queue.push_back(&kern_stmt->root());
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg
