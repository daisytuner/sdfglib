#include "sdfg/passes/symbolic/symbol_promotion.h"

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace passes {

symbolic::Expression SymbolPromotion::
    as_symbol(const data_flow::DataFlowGraph& dataflow, const data_flow::Tasklet& tasklet, const std::string& op) {
    for (auto& iedge : dataflow.in_edges(tasklet)) {
        if (iedge.dst_conn() == op) {
            auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
            if (dynamic_cast<const data_flow::ConstantNode*>(&iedge.src()) != nullptr) {
                int64_t value = helpers::parse_number(src.data());
                return symbolic::integer(value);
            } else {
                return symbolic::symbol(src.data());
            }
        }
    }
    return SymEngine::null;
};

bool SymbolPromotion::can_be_applied(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    data_flow::DataFlowGraph& dataflow
) {
    auto& sdfg = builder.subject();

    // Criterion: Single-tasklet-graph
    for (auto& edge : dataflow.edges()) {
        if (edge.type() != data_flow::MemletType::Computational) {
            return false;
        }
    }
    if (dataflow.tasklets().size() != 1) {
        return false;
    }
    auto tasklet = *dataflow.tasklets().begin();

    // Criterion: Tasklet is not a fp operation
    if (data_flow::is_floating_point(tasklet->code())) {
        return false;
    }

    // Symbolic expressions and operators are per-default
    // interpreted as signed, unless specified otherwise.

    // Criterion: Tasklet is not an unsigned operation
    if (data_flow::is_unsigned(tasklet->code())) {
        return false;
    }

    // Criterion: Inputs are signed integers
    for (auto& iedge : dataflow.in_edges(*tasklet)) {
        if (iedge.subset().size() > 0) {
            return false;
        }

        // Connector type must be a signed integer
        if (!types::is_integer(iedge.base_type().primitive_type()) ||
            types::is_unsigned(iedge.base_type().primitive_type())) {
            return false;
        }

        // No cast on memlet
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        const types::IType* src_type = nullptr;
        if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(&src)) {
            src_type = &const_node->type();
        } else {
            src_type = &sdfg.type(src.data());
        }
        if (*src_type != iedge.base_type()) {
            return false;
        }

        // isolated input to simplify removal
        if (dataflow.in_degree(iedge.src()) > 0) {
            return false;
        }
    }

    auto& oedge = *dataflow.out_edges(*tasklet).begin();
    if (oedge.subset().size() > 0) {
        return false;
    }
    if (!types::is_integer(oedge.base_type().primitive_type()) ||
        types::is_unsigned(oedge.base_type().primitive_type())) {
        return false;
    }

    // No cast on memlet
    auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
    const types::IType* dst_type = nullptr;
    if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(&dst)) {
        dst_type = &const_node->type();
    } else {
        dst_type = &sdfg.type(dst.data());
    }
    if (*dst_type != oedge.base_type()) {
        return false;
    }

    // isolated output
    if (dataflow.out_degree(oedge.dst()) > 0) {
        return false;
    }

    // Furthermore,

    // Criterion: Known tasklet class. To be extended on the go.
    switch (tasklet->code()) {
        case data_flow::TaskletCode::assign:
        case data_flow::TaskletCode::int_add:
        case data_flow::TaskletCode::int_sub:
        case data_flow::TaskletCode::int_mul:
        case data_flow::TaskletCode::int_sdiv:
        case data_flow::TaskletCode::int_srem:
        case data_flow::TaskletCode::int_smin:
        case data_flow::TaskletCode::int_smax:
        case data_flow::TaskletCode::int_abs:
            return true;
        case data_flow::TaskletCode::int_ashr:
        case data_flow::TaskletCode::int_shl: {
            // Shift is constant
            return !tasklet->has_constant_input(1);
        }
        case data_flow::TaskletCode::int_and:
        case data_flow::TaskletCode::int_or:
        case data_flow::TaskletCode::int_xor: {
            // Only for booleans
            for (auto& iedge : dataflow.in_edges(*tasklet)) {
                auto& type = iedge.result_type(sdfg);
                if (type.primitive_type() != types::PrimitiveType::Bool) {
                    return false;
                }
            }
            if (oedge.base_type().primitive_type() != types::PrimitiveType::Bool) {
                return false;
            }
            return true;
        }
        default:
            return false;
    }
};

void SymbolPromotion::apply(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block
) {
    auto& dataflow = block.dataflow();
    auto tasklet = *dataflow.tasklets().begin();
    auto& output_node = (*dataflow.out_edges(*tasklet).begin()).dst();
    auto& access_node = dynamic_cast<data_flow::AccessNode&>(output_node);

    // Build expression
    symbolic::Symbol lhs = sdfg::symbolic::symbol(access_node.data());
    symbolic::Expression rhs;

    switch (tasklet->code()) {
        case data_flow::TaskletCode::assign: {
            rhs = as_symbol(dataflow, *tasklet, tasklet->input(0));
            break;
        }
        case data_flow::TaskletCode::int_add: {
            rhs = symbolic::
                add(as_symbol(dataflow, *tasklet, tasklet->input(0)), as_symbol(dataflow, *tasklet, tasklet->input(1)));
            break;
        }
        case data_flow::TaskletCode::int_sub: {
            rhs = symbolic::
                sub(as_symbol(dataflow, *tasklet, tasklet->input(0)), as_symbol(dataflow, *tasklet, tasklet->input(1)));
            break;
        }
        case data_flow::TaskletCode::int_mul: {
            rhs = symbolic::
                mul(as_symbol(dataflow, *tasklet, tasklet->input(0)), as_symbol(dataflow, *tasklet, tasklet->input(1)));
            break;
        }
        case data_flow::TaskletCode::int_sdiv: {
            rhs = symbolic::
                div(as_symbol(dataflow, *tasklet, tasklet->input(0)), as_symbol(dataflow, *tasklet, tasklet->input(1)));
            break;
        }
        case data_flow::TaskletCode::int_srem: {
            auto op_1 = as_symbol(dataflow, *tasklet, tasklet->input(0));
            auto op_2 = as_symbol(dataflow, *tasklet, tasklet->input(1));
            rhs = symbolic::mod(op_1, op_2);
            break;
        }
        case data_flow::TaskletCode::int_smin: {
            auto op_1 = as_symbol(dataflow, *tasklet, tasklet->input(0));
            auto op_2 = as_symbol(dataflow, *tasklet, tasklet->input(1));
            rhs = symbolic::min(op_1, op_2);
            break;
        }
        case data_flow::TaskletCode::int_smax: {
            auto op_1 = as_symbol(dataflow, *tasklet, tasklet->input(0));
            auto op_2 = as_symbol(dataflow, *tasklet, tasklet->input(1));
            rhs = symbolic::max(op_1, op_2);
            break;
        }
        case data_flow::TaskletCode::int_abs: {
            auto op_1 = as_symbol(dataflow, *tasklet, tasklet->input(0));
            rhs = symbolic::abs(op_1);
            break;
        }
        case data_flow::TaskletCode::int_ashr: {
            rhs = symbolic::
                div(as_symbol(dataflow, *tasklet, tasklet->input(0)),
                    symbolic::pow(symbolic::integer(2), as_symbol(dataflow, *tasklet, tasklet->input(1))));
            break;
        }
        case data_flow::TaskletCode::int_shl: {
            rhs = symbolic::
                mul(as_symbol(dataflow, *tasklet, tasklet->input(0)),
                    symbolic::pow(symbolic::integer(2), as_symbol(dataflow, *tasklet, tasklet->input(1))));
            break;
        }
        case data_flow::TaskletCode::int_and: {
            auto op_1_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(0)), symbolic::__true__());
            auto op_2_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(1)), symbolic::__true__());
            rhs = symbolic::And(op_1_is_true, op_2_is_true);
            break;
        }
        case data_flow::TaskletCode::int_or: {
            auto op_1_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(0)), symbolic::__true__());
            auto op_2_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(1)), symbolic::__true__());
            rhs = symbolic::Or(op_1_is_true, op_2_is_true);
            break;
        }
        case data_flow::TaskletCode::int_xor: {
            auto op_1_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(0)), symbolic::__true__());
            auto op_2_is_true = symbolic::Eq(as_symbol(dataflow, *tasklet, tasklet->input(1)), symbolic::__true__());
            rhs = symbolic::
                And(symbolic::Or(op_1_is_true, op_2_is_true), symbolic::Not(symbolic::And(op_1_is_true, op_2_is_true)));
            break;
        }
        default: {
            throw InvalidSDFGException("SymbolPromotion: Invalid tasklet code");
        }
    }

    // Split states and set transition
    builder.add_block_before(sequence, block, {{lhs, rhs}}, block.debug_info());
    builder.clear_node(block, *tasklet);
};

SymbolPromotion::SymbolPromotion()
    : Pass() {

      };

std::string SymbolPromotion::name() { return "SymbolPromotion"; };

bool SymbolPromotion::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
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
        } else if (auto sloop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&sloop_stmt->root());
        }
    }

    return applied;
};

} // namespace passes
} // namespace sdfg
