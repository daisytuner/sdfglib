#include "sdfg/passes/symbolic/symbol_promotion.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace passes {

symbolic::Expression SymbolPromotion::as_symbol(const data_flow::DataFlowGraph& dataflow,
                                                const data_flow::Tasklet& tasklet,
                                                const std::string& op) {
    if (op.compare(0, 3, "_in") == 0) {
        for (auto& iedge : dataflow.in_edges(tasklet)) {
            if (iedge.dst_conn() == op) {
                auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
                return symbolic::symbol(src.data());
            }
        }
        throw std::invalid_argument("Invalid input connector");
    } else {
        int64_t value = std::stoll(op);
        return symbolic::integer(value);
    }
};

bool SymbolPromotion::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                     analysis::AnalysisManager& analysis_manager,
                                     data_flow::DataFlowGraph& dataflow) {
    // Criterion: Single tasklet in graph
    // Has to run before all the other passes
    if (dataflow.tasklets().size() != 1) {
        return false;
    }

    // Criterion: Tasklet has single output
    auto tasklet = *dataflow.tasklets().begin();
    if (dataflow.out_degree(*tasklet) > 1) {
        return false;
    }
    for (auto& edge : dataflow.edges()) {
        if (edge.src_conn() == "refs" || edge.dst_conn() == "refs") {
            return false;
        }
    }

    // Criterion: Tasklet has only scalar integer inputs and outputs
    for (auto& iedge : dataflow.in_edges(*tasklet)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        auto& type = builder.subject().type(src.data());
        auto scalar = dynamic_cast<const types::Scalar*>(&type);
        if (!scalar || !types::is_integer(scalar->primitive_type())) {
            return false;
        }

        if (dataflow.in_degree(src) > 0) {
            return false;
        }
    }

    for (auto& oedge : dataflow.out_edges(*tasklet)) {
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
        auto& type = builder.subject().type(dst.data());
        auto scalar = dynamic_cast<const types::Scalar*>(&type);
        if (!scalar || !types::is_integer(scalar->primitive_type())) {
            return false;
        }

        if (dataflow.out_degree(dst) > 0) {
            return false;
        }
    }

    // Criterion: Known tasklet class. To be extended on the go.
    switch (tasklet->code()) {
        case data_flow::TaskletCode::assign:
        case data_flow::TaskletCode::add:
        case data_flow::TaskletCode::sub:
        case data_flow::TaskletCode::mul:
        case data_flow::TaskletCode::max:
        case data_flow::TaskletCode::min:
            return true;
        case data_flow::TaskletCode::shift_left: {
            // Shift is constant
            return !tasklet->needs_connector(1);
        }
        case data_flow::TaskletCode::bitwise_and:
        case data_flow::TaskletCode::bitwise_or:
        case data_flow::TaskletCode::bitwise_xor:
        case data_flow::TaskletCode::bitwise_not: {
            // Bitwise operation is constant for boolean inputs
            auto& sdfg = builder.subject();
            for (auto& edge : dataflow.in_edges(*tasklet)) {
                auto& src = dynamic_cast<const data_flow::AccessNode&>(edge.src());
                auto& type = sdfg.type(src.data());
                auto scalar = dynamic_cast<const types::Scalar*>(&type);
                if (scalar && scalar->primitive_type() != types::PrimitiveType::Bool) {
                    return false;
                }
            }
            for (auto& edge : dataflow.out_edges(*tasklet)) {
                auto& dst = dynamic_cast<const data_flow::AccessNode&>(edge.dst());
                auto& type = sdfg.type(dst.data());
                auto scalar = dynamic_cast<const types::Scalar*>(&type);
                if (scalar && scalar->primitive_type() != types::PrimitiveType::Bool) {
                    return false;
                }
            }
            return true;
        }
        default:
            return false;
    }
};

void SymbolPromotion::apply(builder::StructuredSDFGBuilder& builder,
                            analysis::AnalysisManager& analysis_manager,
                            structured_control_flow::Sequence& sequence,
                            structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();
    auto tasklet = *dataflow.tasklets().begin();
    auto& output_node = (*dataflow.out_edges(*tasklet).begin()).dst();
    auto& access_node = dynamic_cast<data_flow::AccessNode&>(output_node);

    // Build expression
    symbolic::Symbol lhs = sdfg::symbolic::symbol(access_node.data());
    symbolic::Expression rhs;

    switch (tasklet->code()) {
        case data_flow::TaskletCode::assign: {
            rhs = as_symbol(dataflow, *tasklet, tasklet->input(0).first);
            break;
        }
        case data_flow::TaskletCode::add: {
            rhs = symbolic::add(as_symbol(dataflow, *tasklet, tasklet->input(0).first),
                                as_symbol(dataflow, *tasklet, tasklet->input(1).first));
            break;
        }
        case data_flow::TaskletCode::sub: {
            rhs = symbolic::sub(as_symbol(dataflow, *tasklet, tasklet->input(0).first),
                                as_symbol(dataflow, *tasklet, tasklet->input(1).first));
            break;
        }
        case data_flow::TaskletCode::mul: {
            rhs = symbolic::mul(as_symbol(dataflow, *tasklet, tasklet->input(0).first),
                                as_symbol(dataflow, *tasklet, tasklet->input(1).first));
            break;
        }
        case data_flow::TaskletCode::max: {
            rhs = symbolic::max(as_symbol(dataflow, *tasklet, tasklet->input(0).first),
                                as_symbol(dataflow, *tasklet, tasklet->input(1).first));
            break;
        }
        case data_flow::TaskletCode::min: {
            rhs = symbolic::min(as_symbol(dataflow, *tasklet, tasklet->input(0).first),
                                as_symbol(dataflow, *tasklet, tasklet->input(1).first));
            break;
        }
        case data_flow::TaskletCode::shift_left: {
            rhs = symbolic::mul(
                as_symbol(dataflow, *tasklet, tasklet->input(0).first),
                symbolic::pow(symbolic::integer(2),
                              as_symbol(dataflow, *tasklet, tasklet->input(1).first)));
            break;
        }
        case data_flow::TaskletCode::bitwise_and: {
            auto op_1_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(0).first),
                                             symbolic::__true__());
            auto op_2_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(1).first),
                                             symbolic::__true__());
            rhs = symbolic::And(op_1_is_true, op_2_is_true);
            break;
        }
        case data_flow::TaskletCode::bitwise_or: {
            auto op_1_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(0).first),
                                             symbolic::__true__());
            auto op_2_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(1).first),
                                             symbolic::__true__());
            rhs = symbolic::Or(op_1_is_true, op_2_is_true);
            break;
        }
        case data_flow::TaskletCode::bitwise_xor: {
            auto op_1_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(0).first),
                                             symbolic::__true__());
            auto op_2_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(1).first),
                                             symbolic::__true__());
            rhs = symbolic::And(symbolic::Or(op_1_is_true, op_2_is_true),
                                symbolic::Not(symbolic::And(op_1_is_true, op_2_is_true)));
            break;
        }
        case data_flow::TaskletCode::bitwise_not: {
            auto op_1_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(0).first),
                                             symbolic::__true__());
            rhs = symbolic::Not(op_1_is_true);
            break;
        }
        default: {
            throw InvalidSDFGException("SymbolPromotion: Invalid tasklet code");
        }
    }

    // Split states and set transition
    auto before = builder.add_block_before(sequence, block);
    before.second.assignments().insert({lhs, rhs});

    builder.clear_node(block, *tasklet);
};

SymbolPromotion::SymbolPromotion()
    : Pass() {

      };

std::string SymbolPromotion::name() { return "SymbolPromotion"; };

bool SymbolPromotion::run_pass(builder::StructuredSDFGBuilder& builder,
                               analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    // Traverse structured SDFG
    std::list<structured_control_flow::ControlFlowNode*> queue = {&builder.subject().root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // If sequence, attempt promotion
        if (auto match = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            size_t i = 0;
            while (i < match->size()) {
                auto entry = match->at(i);
                if (auto block = dynamic_cast<structured_control_flow::Block*>(&entry.first)) {
                    if (can_be_applied(builder, analysis_manager, block->dataflow())) {
                        apply(builder, analysis_manager, *match, *block);
                        applied = true;
                        i++;
                    }
                }
                i++;
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
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            queue.push_back(&for_stmt->root());
        } else if (auto kern_stmt = dynamic_cast<const structured_control_flow::Kernel*>(current)) {
            queue.push_back(&kern_stmt->root());
        }
    }

    return applied;
};

}  // namespace passes
}  // namespace sdfg
