#pragma once

#include "sdfg/data_flow/code_node.h"
#include "sdfg/exceptions.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class FunctionBuilder;
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

enum TaskletCode {
    assign,
    // Floating-point-specific operations
    // Operations
    fp_neg,
    fp_add,
    fp_sub,
    fp_mul,
    fp_div,
    fp_rem,
    fp_fma,
    // Comparisions
    fp_oeq,
    fp_one,
    fp_oge,
    fp_ogt,
    fp_ole,
    fp_olt,
    fp_ord,
    fp_ueq,
    fp_une,
    fp_ugt,
    fp_uge,
    fp_ult,
    fp_ule,
    fp_uno,
    // Integer-specific operations
    // Operations
    int_add,
    int_sub,
    int_mul,
    int_sdiv,
    int_srem,
    int_udiv,
    int_urem,
    int_and,
    int_or,
    int_xor,
    int_shl,
    int_ashr,
    int_lshr,
    int_smin,
    int_smax,
    int_scmp,
    int_umin,
    int_umax,
    int_ucmp,
    // Comparisions
    int_eq,
    int_ne,
    int_sge,
    int_sgt,
    int_sle,
    int_slt,
    int_uge,
    int_ugt,
    int_ule,
    int_ult
};

constexpr size_t arity(TaskletCode c) {
    switch (c) {
        case TaskletCode::assign:
            return 1;
        // Integer Relational Ops
        case TaskletCode::int_add:
        case TaskletCode::int_sub:
        case TaskletCode::int_mul:
        case TaskletCode::int_sdiv:
        case TaskletCode::int_srem:
        case TaskletCode::int_udiv:
        case TaskletCode::int_urem:
        case TaskletCode::int_and:
        case TaskletCode::int_or:
        case TaskletCode::int_xor:
        case TaskletCode::int_shl:
        case TaskletCode::int_ashr:
        case TaskletCode::int_lshr:
        case TaskletCode::int_smin:
        case TaskletCode::int_smax:
        case TaskletCode::int_umin:
        case TaskletCode::int_scmp:
        case TaskletCode::int_umax:
        case TaskletCode::int_ucmp:
            return 2;
        // Comparisions
        case TaskletCode::int_eq:
        case TaskletCode::int_ne:
        case TaskletCode::int_sge:
        case TaskletCode::int_sgt:
        case TaskletCode::int_sle:
        case TaskletCode::int_slt:
        case TaskletCode::int_uge:
        case TaskletCode::int_ugt:
        case TaskletCode::int_ule:
        case TaskletCode::int_ult:
            return 2;
        // Floating Point
        case TaskletCode::fp_neg:
            return 1;
        case TaskletCode::fp_add:
        case TaskletCode::fp_sub:
        case TaskletCode::fp_mul:
        case TaskletCode::fp_div:
        case TaskletCode::fp_rem:
            return 2;
        // Comparisions
        case TaskletCode::fp_oeq:
        case TaskletCode::fp_one:
        case TaskletCode::fp_oge:
        case TaskletCode::fp_ogt:
        case TaskletCode::fp_ole:
        case TaskletCode::fp_olt:
        case TaskletCode::fp_ord:
        case TaskletCode::fp_ueq:
        case TaskletCode::fp_une:
        case TaskletCode::fp_ugt:
        case TaskletCode::fp_uge:
        case TaskletCode::fp_ult:
        case TaskletCode::fp_ule:
        case TaskletCode::fp_uno:
        return 2;
        case TaskletCode::fp_fma:
            return 3;
    };
    throw InvalidSDFGException("Invalid tasklet code");
};

constexpr bool is_unsigned(TaskletCode c) {
    switch (c) {
        case TaskletCode::int_udiv:
        case TaskletCode::int_urem:
        case TaskletCode::int_lshr:
        case TaskletCode::int_umin:
        case TaskletCode::int_umax:
        case TaskletCode::int_ucmp:
        case TaskletCode::int_uge:
        case TaskletCode::int_ugt:
        case TaskletCode::int_ule:
        case TaskletCode::int_ult:
            return true;
        default:
            return false;
    }
};

constexpr bool is_integer(TaskletCode c) {
    switch (c) {
        // Operations
        case TaskletCode::int_add:
        case TaskletCode::int_sub:
        case TaskletCode::int_mul:
        case TaskletCode::int_sdiv:
        case TaskletCode::int_srem:
        case TaskletCode::int_udiv:
        case TaskletCode::int_urem:
        case TaskletCode::int_and:
        case TaskletCode::int_or:
        case TaskletCode::int_xor:
        case TaskletCode::int_shl:
        case TaskletCode::int_ashr:
        case TaskletCode::int_lshr:
        case TaskletCode::int_smin:
        case TaskletCode::int_smax:
        case TaskletCode::int_scmp:
        case TaskletCode::int_umin:
        case TaskletCode::int_umax:
        case TaskletCode::int_ucmp:
        // Comparisions
        case TaskletCode::int_eq:
        case TaskletCode::int_ne:
        case TaskletCode::int_sge:
        case TaskletCode::int_sgt:
        case TaskletCode::int_sle:
        case TaskletCode::int_slt:
        case TaskletCode::int_uge:
        case TaskletCode::int_ugt:
        case TaskletCode::int_ule:
        case TaskletCode::int_ult:
            return true;
        default:
            return false;
    }
}

constexpr bool is_floating_point(TaskletCode c) {
    switch (c) {
        // Operations
        case TaskletCode::fp_neg:
        case TaskletCode::fp_add:
        case TaskletCode::fp_sub:
        case TaskletCode::fp_mul:
        case TaskletCode::fp_div:
        case TaskletCode::fp_rem:
        case TaskletCode::fp_fma:
        // Comparisions
        case TaskletCode::fp_oeq:
        case TaskletCode::fp_one:
        case TaskletCode::fp_oge:
        case TaskletCode::fp_ogt:
        case TaskletCode::fp_ole:
        case TaskletCode::fp_olt:
        case TaskletCode::fp_ord:
        case TaskletCode::fp_ueq:
        case TaskletCode::fp_une:
        case TaskletCode::fp_ugt:
        case TaskletCode::fp_uge:
        case TaskletCode::fp_ult:
        case TaskletCode::fp_ule:
        case TaskletCode::fp_uno:
            return true;
        default:
            return false;
    }
};

class Tasklet : public CodeNode {
    friend class sdfg::builder::FunctionBuilder;
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    TaskletCode code_;

    Tasklet(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const TaskletCode code,
        const std::string& output,
        const std::vector<std::string>& inputs
    );

public:
    Tasklet(const Tasklet& data_node) = delete;
    Tasklet& operator=(const Tasklet&) = delete;

    void validate(const Function& function) const override;

    TaskletCode code() const;

    const std::string& output() const;

    virtual std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};
} // namespace data_flow
} // namespace sdfg
