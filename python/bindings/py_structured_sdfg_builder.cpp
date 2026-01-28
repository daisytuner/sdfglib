#include "py_structured_sdfg_builder.h"
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/symbolic/symbolic.h>
#include <sstream>
#include <symengine/add.h>
#include <symengine/functions.h>
#include <symengine/integer.h>
#include <symengine/logic.h>
#include <symengine/real_double.h>
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/cast_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/transpose_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/passes/debug_info_propagation.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/visualizer/dot_visualizer.h"

using namespace sdfg::structured_control_flow;

PyStructuredSDFGBuilder::PyStructuredSDFGBuilder(const std::string& name)
    : builder(name, sdfg::FunctionType_CPU, sdfg::types::Scalar(sdfg::types::PrimitiveType::Void)) {
    scope_stack.push_back({&builder.subject().root(), nullptr, -1});
}

PyStructuredSDFGBuilder::PyStructuredSDFGBuilder(const std::string& name, const sdfg::types::IType& return_type)
    : builder(name, sdfg::FunctionType_CPU, return_type) {
    scope_stack.push_back({&builder.subject().root(), nullptr, -1});
}

PyStructuredSDFG PyStructuredSDFGBuilder::move() {
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
    sdfg::passes::DebugInfoPropagation debug_info_propagation_pass;
    debug_info_propagation_pass.run(builder, analysis_manager);

    auto sdfg = builder.move();
    return PyStructuredSDFG(sdfg);
}

void PyStructuredSDFGBuilder::add_container(const std::string& name, const sdfg::types::IType& type, bool is_argument) {
    builder.add_container(name, type, is_argument);
}

void PyStructuredSDFGBuilder::
    add_structure(const std::string& name, const std::vector<const sdfg::types::IType*>& member_types) {
    auto defined_structures = builder.subject().structures();
    if (std::find(defined_structures.begin(), defined_structures.end(), name) != defined_structures.end()) {
        return;
    }

    auto& structure_definition = builder.add_structure(name, false);
    for (const auto* member_type : member_types) {
        structure_definition.add_member(*member_type);
    }
}

bool PyStructuredSDFGBuilder::exists(const std::string& name) { return builder.subject().exists(name); }

void PyStructuredSDFGBuilder::set_return_type(const sdfg::types::IType& type) { builder.set_return_type(type); }

std::string PyStructuredSDFGBuilder::get_sizeof(const sdfg::types::IType& type) {
    auto expr = sdfg::symbolic::size_of_type(type);
    return expr->__str__();
}

sdfg::structured_control_flow::Sequence& PyStructuredSDFGBuilder::current_sequence() {
    if (scope_stack.empty()) {
        throw std::runtime_error("Scope stack is empty!");
    }
    return *scope_stack.back().sequence;
}

void PyStructuredSDFGBuilder::add_return(const std::string& data, const sdfg::DebugInfo& debug_info) {
    builder.add_return(current_sequence(), data, {}, debug_info);
}

void PyStructuredSDFGBuilder::
    add_constant_return(const std::string& value, const sdfg::types::IType& type, const sdfg::DebugInfo& debug_info) {
    builder.add_constant_return(current_sequence(), value, type, {}, debug_info);
}

void PyStructuredSDFGBuilder::begin_if(const std::string& condition, const sdfg::DebugInfo& debug_info) {
    auto& parent = current_sequence();
    auto cond_expr = sdfg::symbolic::parse(condition);

    auto cond_bool = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(cond_expr);
    if (cond_bool.is_null()) {
        throw std::runtime_error("Condition must be a boolean expression: " + condition);
    }

    auto& if_node = builder.add_if_else(parent, {}, debug_info);
    auto& then_block = builder.add_case(if_node, cond_bool, debug_info);

    scope_stack.push_back({&then_block, &if_node, 0});
}

void PyStructuredSDFGBuilder::begin_else(const sdfg::DebugInfo& debug_info) {
    auto current = scope_stack.back();
    auto* if_node = dynamic_cast<sdfg::structured_control_flow::IfElse*>(current.node);
    if (!if_node || current.branch_index != 0) {
        throw std::runtime_error("Cannot begin_else: not in an if block or already in else");
    }

    auto cond = if_node->at(0).second;
    auto not_cond = SymEngine::logical_not(cond);

    scope_stack.pop_back();
    auto& else_block = builder.add_case(*if_node, not_cond, debug_info);
    scope_stack.push_back({&else_block, if_node, 1});
}

void PyStructuredSDFGBuilder::end_if() {
    auto current = scope_stack.back();
    auto* if_node = dynamic_cast<sdfg::structured_control_flow::IfElse*>(current.node);
    if (!if_node) {
        throw std::runtime_error("Cannot end_if: not in an if/else block");
    }
    scope_stack.pop_back();
}

void PyStructuredSDFGBuilder::begin_while(const std::string& condition, const sdfg::DebugInfo& debug_info) {
    auto& parent = current_sequence();
    auto& while_node = builder.add_while(parent, {}, debug_info);

    auto& while_body = while_node.root();

    auto cond_expr = sdfg::symbolic::parse(condition);
    auto cond_bool = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(cond_expr);
    if (cond_bool.is_null()) {
        throw std::runtime_error("Condition must be a boolean expression: " + condition);
    }

    auto not_cond = SymEngine::logical_not(cond_bool);

    auto& if_node = builder.add_if_else(while_body, {}, debug_info);
    auto& then_block = builder.add_case(if_node, not_cond, debug_info);

    builder.add_break(then_block, {}, debug_info);

    scope_stack.push_back({&while_body, &while_node, 0});
}

void PyStructuredSDFGBuilder::end_while() {
    auto current = scope_stack.back();
    auto* while_node = dynamic_cast<sdfg::structured_control_flow::While*>(current.node);
    if (!while_node) {
        throw std::runtime_error("Cannot end_while: not in a while loop");
    }
    scope_stack.pop_back();
}

void PyStructuredSDFGBuilder::begin_for(
    const std::string& var,
    const std::string& start,
    const std::string& end,
    const std::string& step,
    const sdfg::DebugInfo& debug_info
) {
    auto& parent = current_sequence();
    auto var_sym = sdfg::symbolic::symbol(var);
    auto start_expr = sdfg::symbolic::parse(start);
    auto end_expr = sdfg::symbolic::parse(end);
    auto step_expr = sdfg::symbolic::parse(step);

    bool is_negative = false;
    if (SymEngine::is_a<SymEngine::Integer>(*step_expr)) {
        auto i = SymEngine::rcp_static_cast<const SymEngine::Integer>(step_expr);
        if (i->is_negative()) is_negative = true;
    } else if (SymEngine::is_a<SymEngine::RealDouble>(*step_expr)) {
        auto d = SymEngine::rcp_static_cast<const SymEngine::RealDouble>(step_expr);
        if (d->as_double() < 0) is_negative = true;
    }

    SymEngine::RCP<const SymEngine::Boolean> condition;
    if (is_negative) {
        condition = SymEngine::Gt(var_sym, end_expr);
    } else {
        condition = SymEngine::Lt(var_sym, end_expr);
    }

    auto update = SymEngine::add(var_sym, step_expr);

    auto& for_node = builder.add_for(parent, var_sym, condition, start_expr, update, {}, debug_info);

    scope_stack.push_back({&for_node.root(), &for_node, 0});
}

void PyStructuredSDFGBuilder::end_for() {
    auto current = scope_stack.back();
    auto* for_node = dynamic_cast<sdfg::structured_control_flow::For*>(current.node);
    if (!for_node) {
        throw std::runtime_error("Cannot end_for: not in a for loop");
    }
    scope_stack.pop_back();
}

void PyStructuredSDFGBuilder::
    add_transition(const std::string& lhs, const std::string& rhs, const sdfg::DebugInfo& debug_info) {
    auto& parent = current_sequence();

    sdfg::symbolic::Symbol lhs_sym = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(sdfg::symbolic::parse(lhs));
    sdfg::symbolic::Expression rhs_sym = sdfg::symbolic::parse(rhs);

    builder.add_block(parent, {{lhs_sym, rhs_sym}}, debug_info);
}

void PyStructuredSDFGBuilder::
    add_assignment(const std::string& target, const std::string& value, const sdfg::DebugInfo& debug_info) {
    auto& parent = current_sequence();
    auto& block = builder.add_block(parent, {}, debug_info);

    auto expr = sdfg::symbolic::parse(value);

    // Parse target
    std::string target_name = target;
    std::vector<SymEngine::RCP<const SymEngine::Basic>> target_indices;

    size_t open_paren = target.find('(');
    if (open_paren != std::string::npos) {
        target_name = target.substr(0, open_paren);

        // Find matching closing parenthesis
        size_t close_paren = std::string::npos;
        int balance = 0;
        for (size_t i = open_paren; i < target.length(); ++i) {
            if (target[i] == '(')
                balance++;
            else if (target[i] == ')')
                balance--;

            if (balance == 0) {
                close_paren = i;
                break;
            }
        }

        if (close_paren == std::string::npos) throw std::runtime_error("Invalid target format: unbalanced parentheses");
        std::string idx_str = target.substr(open_paren + 1, close_paren - open_paren - 1);
        target_indices.push_back(sdfg::symbolic::parse(idx_str));
    }

    // Get target type
    if (!builder.subject().exists(target_name)) {
        throw std::runtime_error("Target container not found: " + target_name);
    }
    auto& target_container_type = builder.subject().type(target_name);
    auto& dst = builder.add_access(block, target_name, debug_info);

    // Determine element type for opcode selection
    const sdfg::types::IType* elem_type = &target_container_type;
    if (target_container_type.type_id() == sdfg::types::TypeID::Pointer) {
        elem_type = &dynamic_cast<const sdfg::types::Pointer&>(target_container_type).pointee_type();
    } else if (target_container_type.type_id() == sdfg::types::TypeID::Array) {
        elem_type = &dynamic_cast<const sdfg::types::Array&>(target_container_type).element_type();
    }

    auto create_source_memlet = [&](const std::string& name, sdfg::data_flow::Tasklet& tasklet, const std::string& conn
                                ) {
        std::string src_name = name;
        std::vector<SymEngine::RCP<const SymEngine::Basic>> src_indices;

        size_t open_paren = name.find('(');
        if (open_paren != std::string::npos) {
            src_name = name.substr(0, open_paren);

            // Find matching closing parenthesis
            size_t close_paren = std::string::npos;
            int balance = 0;
            for (size_t i = open_paren; i < name.length(); ++i) {
                if (name[i] == '(')
                    balance++;
                else if (name[i] == ')')
                    balance--;

                if (balance == 0) {
                    close_paren = i;
                    break;
                }
            }

            if (close_paren != std::string::npos) {
                std::string idx_str = name.substr(open_paren + 1, close_paren - open_paren - 1);
                src_indices.push_back(sdfg::symbolic::parse(idx_str));
            }
        }

        if (builder.subject().exists(src_name)) {
            auto& src = builder.add_access(block, src_name, debug_info);
            auto& src_type = builder.subject().type(src_name);

            const sdfg::types::IType* src_memlet_type = &src_type;
            sdfg::types::Scalar ptr_scalar(sdfg::types::PrimitiveType::UInt64);
            if (src_type.type_id() == sdfg::types::TypeID::Pointer && src_indices.empty()) {
                src_memlet_type = &ptr_scalar;
            }

            builder.add_computational_memlet(block, src, tasklet, conn, src_indices, *src_memlet_type, debug_info);
        } else {
            auto& src = builder.add_constant(block, name, *elem_type, debug_info);
            builder.add_computational_memlet(block, src, tasklet, conn, {}, *elem_type, debug_info);
        }
    };

    // 1. Assignment (s = 0 or s = x or A[i] = x)
    if (SymEngine::is_a<SymEngine::Integer>(*expr) || SymEngine::is_a<SymEngine::RealDouble>(*expr) ||
        SymEngine::is_a<SymEngine::Symbol>(*expr) || SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        std::string val_str = expr->__str__();
        auto& tasklet = builder.add_tasklet(block, sdfg::data_flow::assign, "_out", {"_in"}, debug_info);

        create_source_memlet(val_str, tasklet, "_in");

        const sdfg::types::IType* memlet_type = &target_container_type;
        sdfg::types::Scalar ptr_scalar(sdfg::types::PrimitiveType::UInt64);
        if (target_container_type.type_id() == sdfg::types::TypeID::Pointer && target_indices.empty()) {
            memlet_type = &ptr_scalar;
        }

        builder.add_computational_memlet(block, tasklet, "_out", dst, target_indices, *memlet_type, debug_info);
    }
    // 2. Addition (s = s + i) or Subtraction (s = s - i)
    else if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        auto args = add->get_args();
        if (args.size() != 2) throw std::runtime_error("Only binary add/sub supported");

        std::string op1 = args[0]->__str__();
        std::string op2 = args[1]->__str__();

        sdfg::data_flow::TaskletCode opcode = sdfg::data_flow::int_add;
        bool is_float =
            (elem_type->primitive_type() == sdfg::types::PrimitiveType::Double ||
             elem_type->primitive_type() == sdfg::types::PrimitiveType::Float);

        if (is_float) opcode = sdfg::data_flow::fp_add;

        // Check for subtraction: a + (-1)*b
        if (SymEngine::is_a<SymEngine::Mul>(*args[0]) || SymEngine::is_a<SymEngine::Mul>(*args[1])) {
            // Check if one operand is -1 * symbol
            auto check_neg = [](const SymEngine::RCP<const SymEngine::Basic>& node, std::string& sym_name) -> bool {
                if (SymEngine::is_a<SymEngine::Mul>(*node)) {
                    auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(node);
                    auto margs = mul->get_args();
                    if (margs.size() == 2 && SymEngine::eq(*margs[0], *SymEngine::integer(-1))) {
                        sym_name = margs[1]->__str__();
                        return true;
                    }
                    // Handle -1.0 for floats
                    if (margs.size() == 2 && SymEngine::is_a<SymEngine::RealDouble>(*margs[0])) {
                        auto d = SymEngine::rcp_static_cast<const SymEngine::RealDouble>(margs[0]);
                        if (d->as_double() == -1.0) {
                            sym_name = margs[1]->__str__();
                            return true;
                        }
                    }
                }
                return false;
            };

            std::string neg_op;
            if (check_neg(args[0], neg_op)) {
                // (-b) + a -> a - b
                op1 = op2;
                op2 = neg_op;
                if (is_float)
                    opcode = sdfg::data_flow::fp_sub;
                else
                    opcode = sdfg::data_flow::int_sub;
            } else if (check_neg(args[1], neg_op)) {
                // a + (-b) -> a - b
                op2 = neg_op;
                if (is_float)
                    opcode = sdfg::data_flow::fp_sub;
                else
                    opcode = sdfg::data_flow::int_sub;
            }
        }

        auto& tasklet = builder.add_tasklet(block, opcode, "_out", {"_in1", "_in2"}, debug_info);

        create_source_memlet(op1, tasklet, "_in1");
        create_source_memlet(op2, tasklet, "_in2");
        builder.add_computational_memlet(block, tasklet, "_out", dst, target_indices, target_container_type, debug_info);
    }
    // 3. Multiplication
    else if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(expr);
        auto args = mul->get_args();
        if (args.size() != 2) throw std::runtime_error("Only binary mul supported");

        std::string op1 = args[0]->__str__();
        std::string op2 = args[1]->__str__();

        sdfg::data_flow::TaskletCode opcode = sdfg::data_flow::int_mul;
        if (elem_type->primitive_type() == sdfg::types::PrimitiveType::Double ||
            elem_type->primitive_type() == sdfg::types::PrimitiveType::Float) {
            opcode = sdfg::data_flow::fp_mul;
        }

        // Check for division
        if (SymEngine::is_a<SymEngine::Pow>(*args[1])) {
            auto pow = SymEngine::rcp_static_cast<const SymEngine::Pow>(args[1]);
            auto pargs = pow->get_args();
            if (SymEngine::eq(*pargs[1], *SymEngine::integer(-1))) {
                op2 = pargs[0]->__str__();
                if (opcode == sdfg::data_flow::fp_mul)
                    opcode = sdfg::data_flow::fp_div;
                else
                    opcode = sdfg::data_flow::int_sdiv;
            }
        } else if (SymEngine::is_a<SymEngine::Pow>(*args[0])) {
            auto pow = SymEngine::rcp_static_cast<const SymEngine::Pow>(args[0]);
            auto pargs = pow->get_args();
            if (SymEngine::eq(*pargs[1], *SymEngine::integer(-1))) {
                // a^-1 * b -> b / a
                std::string tmp = op1;
                op1 = op2;
                op2 = pargs[0]->__str__();
                if (opcode == sdfg::data_flow::fp_mul)
                    opcode = sdfg::data_flow::fp_div;
                else
                    opcode = sdfg::data_flow::int_sdiv;
            }
        }

        auto& tasklet = builder.add_tasklet(block, opcode, "_out", {"_in1", "_in2"}, debug_info);

        create_source_memlet(op1, tasklet, "_in1");
        create_source_memlet(op2, tasklet, "_in2");
        builder.add_computational_memlet(block, tasklet, "_out", dst, target_indices, target_container_type, debug_info);
    } else {
        throw std::runtime_error("Unsupported assignment expression: " + value);
    }
}

size_t PyStructuredSDFGBuilder::add_block(const sdfg::DebugInfo& debug_info) {
    auto& parent = current_sequence();
    auto& block = builder.add_block(parent, {}, debug_info);
    return reinterpret_cast<size_t>(&block);
}

size_t PyStructuredSDFGBuilder::add_access(size_t block_ptr, const std::string& name, const sdfg::DebugInfo& debug_info) {
    auto* block = reinterpret_cast<sdfg::structured_control_flow::Block*>(block_ptr);
    auto& access = builder.add_access(*block, name, debug_info);
    return reinterpret_cast<size_t>(&access);
}

size_t PyStructuredSDFGBuilder::add_constant(
    size_t block_ptr, const std::string& value, const sdfg::types::IType& type, const sdfg::DebugInfo& debug_info
) {
    auto* block = reinterpret_cast<sdfg::structured_control_flow::Block*>(block_ptr);
    auto& constant = builder.add_constant(*block, value, type, debug_info);
    return reinterpret_cast<size_t>(&constant);
}

size_t PyStructuredSDFGBuilder::add_tasklet(
    size_t block_ptr,
    sdfg::data_flow::TaskletCode code,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const sdfg::DebugInfo& debug_info
) {
    auto* block = reinterpret_cast<sdfg::structured_control_flow::Block*>(block_ptr);
    if (outputs.empty()) throw std::runtime_error("Tasklet must have at least one output");
    auto& tasklet = builder.add_tasklet(*block, code, outputs[0], inputs, debug_info);
    return reinterpret_cast<size_t>(&tasklet);
}

void PyStructuredSDFGBuilder::add_memlet(
    size_t block_ptr,
    size_t src_ptr,
    const std::string& src_conn,
    size_t dst_ptr,
    const std::string& dst_conn,
    const std::string& subset,
    const sdfg::types::IType* type_arg,
    const sdfg::DebugInfo& debug_info
) {
    auto* block = reinterpret_cast<sdfg::structured_control_flow::Block*>(block_ptr);
    auto* src = reinterpret_cast<sdfg::data_flow::DataFlowNode*>(src_ptr);
    auto* dst = reinterpret_cast<sdfg::data_flow::DataFlowNode*>(dst_ptr);

    std::vector<SymEngine::RCP<const SymEngine::Basic>> indices;
    if (!subset.empty()) {
        std::stringstream ss(subset);
        std::string segment;
        while (std::getline(ss, segment, ',')) {
            indices.push_back(sdfg::symbolic::parse(segment));
        }
    }

    const sdfg::types::IType* type = type_arg;

    if (!type) {
        if (auto* constant = dynamic_cast<sdfg::data_flow::ConstantNode*>(src)) {
            type = &constant->type();
        } else if (auto* access = dynamic_cast<sdfg::data_flow::AccessNode*>(src)) {
            type = &builder.subject().type(access->data());
        } else if (auto* access = dynamic_cast<sdfg::data_flow::AccessNode*>(dst)) {
            type = &builder.subject().type(access->data());
        }
    }
    if (!type) {
        throw std::runtime_error("Could not determine type for memlet (neither src nor dst is AccessNode/ConstantNode)"
        );
    }

    if (auto* t_src = dynamic_cast<sdfg::data_flow::Tasklet*>(src)) {
        if (auto* a_dst = dynamic_cast<sdfg::data_flow::AccessNode*>(dst)) {
            builder.add_computational_memlet(*block, *t_src, src_conn, *a_dst, indices, *type, debug_info);
            return;
        }
    }
    if (auto* l_src = dynamic_cast<sdfg::data_flow::LibraryNode*>(src)) {
        if (auto* a_dst = dynamic_cast<sdfg::data_flow::AccessNode*>(dst)) {
            builder.add_computational_memlet(*block, *l_src, src_conn, *a_dst, indices, *type, debug_info);
            return;
        }
    }

    if (auto* a_src = dynamic_cast<sdfg::data_flow::AccessNode*>(src)) {
        if (auto* t_dst = dynamic_cast<sdfg::data_flow::Tasklet*>(dst)) {
            builder.add_computational_memlet(*block, *a_src, *t_dst, dst_conn, indices, *type, debug_info);
            return;
        }
        if (auto* l_dst = dynamic_cast<sdfg::data_flow::LibraryNode*>(dst)) {
            builder.add_computational_memlet(*block, *a_src, *l_dst, dst_conn, indices, *type, debug_info);
            return;
        }
    }

    throw std::runtime_error("Unsupported memlet connection (must be Access<->Tasklet/LibraryNode)");
}

void PyStructuredSDFGBuilder::add_reference_memlet(
    size_t block_ptr,
    size_t src_ptr,
    size_t dst_ptr,
    const std::string& subset,
    const sdfg::types::IType* type_arg,
    const sdfg::DebugInfo& debug_info
) {
    auto* block = reinterpret_cast<sdfg::structured_control_flow::Block*>(block_ptr);
    auto* src = reinterpret_cast<sdfg::data_flow::AccessNode*>(src_ptr);
    auto* dst = reinterpret_cast<sdfg::data_flow::AccessNode*>(dst_ptr);

    std::vector<SymEngine::RCP<const SymEngine::Basic>> indices;
    if (!subset.empty()) {
        std::stringstream ss(subset);
        std::string segment;
        while (std::getline(ss, segment, ',')) {
            indices.push_back(sdfg::symbolic::parse(segment));
        }
    }

    const sdfg::types::IType* type = type_arg;

    if (!type) {
        if (auto* constant = dynamic_cast<sdfg::data_flow::ConstantNode*>(src)) {
            type = &constant->type();
        } else if (auto* access = dynamic_cast<sdfg::data_flow::AccessNode*>(src)) {
            type = &builder.subject().type(access->data());
        } else if (auto* access = dynamic_cast<sdfg::data_flow::AccessNode*>(dst)) {
            type = &builder.subject().type(access->data());
        }
    }
    if (!type) {
        throw std::runtime_error("Could not determine type for memlet (neither src nor dst is AccessNode/ConstantNode)"
        );
    }

    builder.add_reference_memlet(*block, *src, *dst, indices, *type, debug_info);
}

size_t PyStructuredSDFGBuilder::add_cmath(size_t block_ptr, const std::string& name, const sdfg::DebugInfo& debug_info) {
    auto* block = reinterpret_cast<sdfg::structured_control_flow::Block*>(block_ptr);
    auto& node = builder.add_library_node<sdfg::math::cmath::CMathNode>(
        *block, debug_info, sdfg::math::cmath::string_to_cmath_function(name), sdfg::types::PrimitiveType::Double
    );
    return reinterpret_cast<size_t>(&node);
}

size_t PyStructuredSDFGBuilder::add_malloc(size_t block_ptr, const std::string& size, const sdfg::DebugInfo& debug_info) {
    auto* block = reinterpret_cast<sdfg::structured_control_flow::Block*>(block_ptr);
    auto size_expr = sdfg::symbolic::parse(size);
    auto& node = builder.add_library_node<sdfg::stdlib::MallocNode>(*block, debug_info, size_expr);
    return reinterpret_cast<size_t>(&node);
}

size_t PyStructuredSDFGBuilder::
    add_memset(size_t block_ptr, const std::string& value, const std::string& num, const sdfg::DebugInfo& debug_info) {
    auto* block = reinterpret_cast<sdfg::structured_control_flow::Block*>(block_ptr);
    auto value_expr = sdfg::symbolic::parse(value);
    auto num_expr = sdfg::symbolic::parse(num);
    auto& node = builder.add_library_node<sdfg::stdlib::MemsetNode>(*block, debug_info, value_expr, num_expr);
    return reinterpret_cast<size_t>(&node);
}

void PyStructuredSDFGBuilder::add_gemm(
    const std::string& A,
    const std::string& B,
    const std::string& C,
    const std::string& alpha,
    const std::string& beta,
    const std::string& m,
    const std::string& n,
    const std::string& k,
    bool trans_a,
    bool trans_b,
    const std::vector<std::string>& a_subset,
    const std::vector<std::string>& b_subset,
    const std::vector<std::string>& c_subset,
    const std::string& lda,
    const std::string& ldb,
    const std::string& ldc,
    const sdfg::DebugInfo& debug_info
) {
    auto& parent = current_sequence();
    auto& view_block = builder.add_block(parent, {}, debug_info);
    auto& block = builder.add_block(parent, {}, debug_info);

    auto sym_m = sdfg::symbolic::parse(m);
    auto sym_n = sdfg::symbolic::parse(n);
    auto sym_k = sdfg::symbolic::parse(k);

    auto sym_lda = lda.empty() ? (trans_a ? sym_m : sym_k) : sdfg::symbolic::parse(lda);
    auto sym_ldb = ldb.empty() ? (trans_b ? sym_k : sym_n) : sdfg::symbolic::parse(ldb);
    auto sym_ldc = ldc.empty() ? sym_n : sdfg::symbolic::parse(ldc);

    auto layout = sdfg::math::blas::BLAS_Layout::RowMajor;
    auto ta = trans_a ? sdfg::math::blas::BLAS_Transpose::Trans : sdfg::math::blas::BLAS_Transpose::No;
    auto tb = trans_b ? sdfg::math::blas::BLAS_Transpose::Trans : sdfg::math::blas::BLAS_Transpose::No;

    auto precision = sdfg::math::blas::BLAS_Precision::d;

    auto& gemm_node = builder.add_library_node<sdfg::math::blas::GEMMNode>(
        block,
        debug_info,
        sdfg::math::blas::ImplementationType_BLAS,
        precision,
        layout,
        ta,
        tb,
        sym_m,
        sym_n,
        sym_k,
        sym_lda,
        sym_ldb,
        sym_ldc
    );

    auto ptr_double = sdfg::types::Pointer(sdfg::types::Scalar(sdfg::types::PrimitiveType::Double));

    auto handle_access =
        [&](const std::string& name, const std::vector<std::string>& subset, const std::string& port, bool is_output) {
            if (subset.empty()) {
                auto& origin = builder.add_access(block, name, debug_info);
                if (is_output)
                    builder.add_computational_memlet(block, gemm_node, port, origin, {}, ptr_double, debug_info);
                else
                    builder.add_computational_memlet(block, origin, gemm_node, port, {}, ptr_double, debug_info);
            } else {
                std::string view_name = builder.find_new_name(name + "_view_");
                builder.add_container(view_name, ptr_double, false);
                auto& view = builder.add_access(view_block, view_name, debug_info);

                sdfg::data_flow::Subset s;
                for (const auto& str : subset) {
                    s.push_back(sdfg::symbolic::parse(str));
                }

                auto& origin = builder.add_access(view_block, name, debug_info);
                builder.add_reference_memlet(view_block, origin, view, s, ptr_double, debug_info);
                if (is_output) {
                    auto& view2 = builder.add_access(block, view_name, debug_info);
                    builder.add_computational_memlet(block, gemm_node, port, view2, {}, ptr_double, debug_info);
                } else {
                    auto& view2 = builder.add_access(block, view_name, debug_info);
                    builder.add_computational_memlet(block, view2, gemm_node, port, {}, ptr_double, debug_info);
                }
            }
        };

    handle_access(A, a_subset, "__A", false);
    handle_access(B, b_subset, "__B", false);
    handle_access(C, c_subset, "__C", false);
    handle_access(C, c_subset, "__C", true);

    auto handle_scalar = [&](const std::string& val, const std::string& port) {
        try {
            size_t idx;
            std::stod(val, &idx);
            if (idx == val.length()) {
                auto& node =
                    builder
                        .add_constant(block, val, sdfg::types::Scalar(sdfg::types::PrimitiveType::Double), debug_info);
                builder.add_computational_memlet(
                    block, node, gemm_node, port, {}, sdfg::types::Scalar(sdfg::types::PrimitiveType::Double), debug_info
                );
                return;
            }
        } catch (...) {
        }
        auto& node = builder.add_access(block, val, debug_info);
        builder.add_computational_memlet(
            block, node, gemm_node, port, {}, sdfg::types::Scalar(sdfg::types::PrimitiveType::Double), debug_info
        );
    };

    handle_scalar(alpha, "__alpha");
    handle_scalar(beta, "__beta");
}

void PyStructuredSDFGBuilder::add_dot(
    const std::string& X,
    const std::string& Y,
    const std::string& result,
    const std::string& n,
    const std::string& incx,
    const std::string& incy,
    const std::vector<std::string>& x_subset,
    const std::vector<std::string>& y_subset,
    const sdfg::DebugInfo& debug_info
) {
    auto& parent = current_sequence();
    auto& block = builder.add_block(parent, {}, debug_info);

    auto sym_n = sdfg::symbolic::parse(n);
    auto sym_incx = sdfg::symbolic::parse(incx);
    auto sym_incy = sdfg::symbolic::parse(incy);

    auto precision = sdfg::math::blas::BLAS_Precision::d;

    auto& dot_node = builder.add_library_node<sdfg::math::blas::DotNode>(
        block, debug_info, sdfg::math::blas::ImplementationType_BLAS, precision, sym_n, sym_incx, sym_incy
    );

    auto ptr_double = sdfg::types::Pointer(sdfg::types::Scalar(sdfg::types::PrimitiveType::Double));

    auto handle_input = [&](const std::string& name, const std::vector<std::string>& subset, const std::string& port) {
        auto& origin = builder.add_access(block, name, debug_info);
        if (subset.empty()) {
            builder.add_computational_memlet(block, origin, dot_node, port, {}, ptr_double, debug_info);
        } else {
            std::string view_name = builder.find_new_name(name + "_view_");
            builder.add_container(view_name, ptr_double, false);
            auto& view = builder.add_access(block, view_name, debug_info);

            sdfg::data_flow::Subset s;
            for (const auto& str : subset) {
                s.push_back(sdfg::symbolic::parse(str));
            }

            builder.add_reference_memlet(block, origin, view, s, ptr_double, debug_info);
            builder.add_computational_memlet(block, view, dot_node, port, {}, ptr_double, debug_info);
        }
    };

    handle_input(X, x_subset, "__x");
    handle_input(Y, y_subset, "__y");

    auto& node_res = builder.add_access(block, result, debug_info);
    builder.add_computational_memlet(
        block, dot_node, "__out", node_res, {}, sdfg::types::Scalar(sdfg::types::PrimitiveType::Double), debug_info
    );
}

void PyStructuredSDFGBuilder::add_elementwise_op(
    const std::string& op_type,
    const std::string& A,
    const std::string& B,
    const std::string& C,
    const std::vector<std::string>& shape_strs,
    const sdfg::DebugInfo& debug_info
) {
    auto& parent = current_sequence();
    auto& block = builder.add_block(parent, {}, debug_info);

    std::vector<sdfg::symbolic::Expression> shape;
    for (const auto& s : shape_strs) {
        shape.push_back(sdfg::symbolic::parse(s));
    }

    sdfg::math::tensor::ElementWiseBinaryNode* node = nullptr;
    if (op_type == "add") {
        node = static_cast<
            sdfg::math::tensor::ElementWiseBinaryNode*>(&builder.add_library_node<
                                                         sdfg::math::tensor::AddNode>(block, debug_info, shape));
    } else if (op_type == "sub") {
        node = static_cast<
            sdfg::math::tensor::ElementWiseBinaryNode*>(&builder.add_library_node<
                                                         sdfg::math::tensor::SubNode>(block, debug_info, shape));
    } else if (op_type == "mul") {
        node = static_cast<
            sdfg::math::tensor::ElementWiseBinaryNode*>(&builder.add_library_node<
                                                         sdfg::math::tensor::MulNode>(block, debug_info, shape));
    } else if (op_type == "div") {
        node = static_cast<
            sdfg::math::tensor::ElementWiseBinaryNode*>(&builder.add_library_node<
                                                         sdfg::math::tensor::DivNode>(block, debug_info, shape));
    } else if (op_type == "pow") {
        node = static_cast<
            sdfg::math::tensor::ElementWiseBinaryNode*>(&builder.add_library_node<
                                                         sdfg::math::tensor::PowNode>(block, debug_info, shape));
    } else if (op_type == "min") {
        node = static_cast<
            sdfg::math::tensor::ElementWiseBinaryNode*>(&builder.add_library_node<
                                                         sdfg::math::tensor::MinimumNode>(block, debug_info, shape));
    } else if (op_type == "max") {
        node = static_cast<
            sdfg::math::tensor::ElementWiseBinaryNode*>(&builder.add_library_node<
                                                         sdfg::math::tensor::MaximumNode>(block, debug_info, shape));
    } else {
        throw std::runtime_error("Unsupported elementwise op: " + op_type);
    }

    auto add_input = [&](const std::string& name, const std::string& conn) {
        if (builder.subject().exists(name)) {
            auto& node_in = builder.add_access(block, name, debug_info);
            auto& type_in = builder.subject().type(name);
            builder.add_computational_memlet(block, node_in, *node, conn, {}, type_in, debug_info);
        } else {
            auto& node_in =
                builder.add_constant(block, name, sdfg::types::Scalar(sdfg::types::PrimitiveType::Double), debug_info);
            builder.add_memlet(
                block,
                node_in,
                "void",
                *node,
                conn,
                {},
                sdfg::types::Scalar(sdfg::types::PrimitiveType::Double),
                debug_info
            );
        }
    };

    add_input(A, "A");
    add_input(B, "B");

    auto& node_c = builder.add_access(block, C, debug_info);
    auto& type_c = builder.subject().type(C);
    builder.add_computational_memlet(block, *node, "C", node_c, {}, type_c, debug_info);
}

void PyStructuredSDFGBuilder::add_elementwise_unary_op(
    const std::string& op_type,
    const std::string& A,
    const std::string& C,
    const std::vector<std::string>& shape_strs,
    const sdfg::DebugInfo& debug_info
) {
    auto& parent = current_sequence();
    auto& block = builder.add_block(parent, {}, debug_info);

    std::vector<sdfg::symbolic::Expression> shape;
    for (const auto& s : shape_strs) {
        shape.push_back(sdfg::symbolic::parse(s));
    }

    sdfg::math::tensor::ElementWiseUnaryNode* node = nullptr;
    if (op_type == "abs") {
        node = static_cast<
            sdfg::math::tensor::ElementWiseUnaryNode*>(&builder.add_library_node<
                                                        sdfg::math::tensor::AbsNode>(block, debug_info, shape));
    } else if (op_type == "sqrt") {
        node = static_cast<
            sdfg::math::tensor::ElementWiseUnaryNode*>(&builder.add_library_node<
                                                        sdfg::math::tensor::SqrtNode>(block, debug_info, shape));
    } else if (op_type == "tanh") {
        node = static_cast<
            sdfg::math::tensor::ElementWiseUnaryNode*>(&builder.add_library_node<
                                                        sdfg::math::tensor::TanhNode>(block, debug_info, shape));
    } else if (op_type == "exp") {
        node = static_cast<
            sdfg::math::tensor::ElementWiseUnaryNode*>(&builder.add_library_node<
                                                        sdfg::math::tensor::ExpNode>(block, debug_info, shape));
    } else {
        throw std::runtime_error("Unsupported elementwise unary op: " + op_type);
    }

    auto add_input = [&](const std::string& name, const std::string& conn) {
        if (builder.subject().exists(name)) {
            auto& node_in = builder.add_access(block, name, debug_info);
            auto& type_in = builder.subject().type(name);
            builder.add_computational_memlet(block, node_in, *node, conn, {}, type_in, debug_info);
        } else {
            auto& node_in =
                builder.add_constant(block, name, sdfg::types::Scalar(sdfg::types::PrimitiveType::Double), debug_info);
            builder.add_memlet(
                block,
                node_in,
                "void",
                *node,
                conn,
                {},
                sdfg::types::Scalar(sdfg::types::PrimitiveType::Double),
                debug_info
            );
        }
    };

    add_input(A, "X");

    auto& node_c = builder.add_access(block, C, debug_info);
    auto& type_c = builder.subject().type(C);
    builder.add_computational_memlet(block, *node, "Y", node_c, {}, type_c, debug_info);
}

void PyStructuredSDFGBuilder::add_transpose(
    const std::string& A,
    const std::string& C,
    const std::vector<std::string>& shape_strs,
    const std::vector<int64_t>& perm,
    const sdfg::DebugInfo& debug_info
) {
    auto& parent = current_sequence();
    auto& block = builder.add_block(parent, {}, debug_info);

    std::vector<sdfg::symbolic::Expression> shape;
    for (const auto& s : shape_strs) {
        shape.push_back(sdfg::symbolic::parse(s));
    }

    auto& node = builder.add_library_node<sdfg::math::tensor::TransposeNode>(block, debug_info, shape, perm);

    auto add_input = [&](const std::string& name, const std::string& conn) {
        if (builder.subject().exists(name)) {
            auto& node_in = builder.add_access(block, name, debug_info);
            auto& type_in = builder.subject().type(name);
            builder.add_computational_memlet(block, node_in, node, conn, {}, type_in, debug_info);
        } else {
            auto& node_in =
                builder.add_constant(block, name, sdfg::types::Scalar(sdfg::types::PrimitiveType::Double), debug_info);
            builder.add_memlet(
                block, node_in, "void", node, conn, {}, sdfg::types::Scalar(sdfg::types::PrimitiveType::Double), debug_info
            );
        }
    };

    add_input(A, "X");

    auto& node_c = builder.add_access(block, C, debug_info);
    auto& type_c = builder.subject().type(C);
    builder.add_computational_memlet(block, node, "Y", node_c, {}, type_c, debug_info);
}

void PyStructuredSDFGBuilder::add_conv(
    const std::string& X,
    const std::string& W,
    const std::string& Y,
    const std::vector<std::string>& shape_strs,
    const std::vector<std::string>& kernel_shape_strs,
    const std::vector<std::string>& strides_strs,
    const std::vector<std::string>& pads_strs,
    const std::vector<std::string>& dilations_strs,
    const std::string& output_channels_str,
    const std::string& group_str,
    const sdfg::DebugInfo& debug_info
) {
    auto& parent = current_sequence();
    auto& block = builder.add_block(parent, {}, debug_info);

    auto transform_dims = [](const std::vector<std::string>& strs) {
        std::vector<sdfg::symbolic::Expression> exprs;
        for (const auto& s : strs) exprs.push_back(sdfg::symbolic::parse(s));
        return exprs;
    };

    auto shape = transform_dims(shape_strs);
    auto kernel_shape = transform_dims(kernel_shape_strs);
    auto strides = transform_dims(strides_strs);
    auto pads = transform_dims(pads_strs);
    auto dilations = transform_dims(dilations_strs);
    auto output_channels = sdfg::symbolic::parse(output_channels_str);
    auto group = sdfg::symbolic::parse(group_str);

    auto& conv_node = builder.add_library_node<sdfg::math::tensor::ConvNode>(
        block, debug_info, shape, kernel_shape, strides, pads, dilations, output_channels, group
    );

    auto add_input = [&](const std::string& name, const std::string& conn) {
        if (builder.subject().exists(name)) {
            auto& node_in = builder.add_access(block, name, debug_info);
            auto& type_in = builder.subject().type(name);
            builder.add_computational_memlet(block, node_in, conv_node, conn, {}, type_in, debug_info);
        } else {
            // Handle constants if needed, usually tensors are variables
            throw std::runtime_error("ConvNode input must be a variable: " + name);
        }
    };

    add_input(X, "X");
    add_input(W, "W");

    // Output
    auto& node_out = builder.add_access(block, Y, debug_info);
    auto& type_out = builder.subject().type(Y);
    builder.add_computational_memlet(block, conv_node, "Y", node_out, {}, type_out, debug_info);
}

void PyStructuredSDFGBuilder::add_cast_op(
    const std::string& A,
    const std::string& C,
    const std::vector<std::string>& shape_strs,
    sdfg::types::PrimitiveType target_type,
    const sdfg::DebugInfo& debug_info
) {
    auto& parent = current_sequence();
    auto& block = builder.add_block(parent, {}, debug_info);

    std::vector<sdfg::symbolic::Expression> shape;
    for (const auto& s : shape_strs) {
        shape.push_back(sdfg::symbolic::parse(s));
    }

    auto& node = builder.add_library_node<sdfg::math::tensor::CastNode>(block, debug_info, shape, target_type);

    auto add_input = [&](const std::string& name, const std::string& conn) {
        if (builder.subject().exists(name)) {
            auto& node_in = builder.add_access(block, name, debug_info);
            auto& type_in = builder.subject().type(name);
            builder.add_computational_memlet(block, node_in, node, conn, {}, type_in, debug_info);
        } else {
            auto& node_in =
                builder.add_constant(block, name, sdfg::types::Scalar(sdfg::types::PrimitiveType::Double), debug_info);
            builder.add_memlet(
                block, node_in, "void", node, conn, {}, sdfg::types::Scalar(sdfg::types::PrimitiveType::Double), debug_info
            );
        }
    };

    add_input(A, "X");

    auto& node_c = builder.add_access(block, C, debug_info);
    auto& type_c = builder.subject().type(C);
    builder.add_computational_memlet(block, node, "Y", node_c, {}, type_c, debug_info);
}

void PyStructuredSDFGBuilder::add_reduce_op(
    const std::string& op_type,
    const std::string& input,
    const std::string& output,
    const std::vector<std::string>& input_shape,
    const std::vector<int64_t>& axes,
    bool keepdims,
    const sdfg::DebugInfo& debug_info
) {
    auto& parent = current_sequence();
    auto& block = builder.add_block(parent, {}, debug_info);

    std::vector<sdfg::symbolic::Expression> shape_exprs;
    for (const auto& s : input_shape) {
        shape_exprs.push_back(sdfg::symbolic::parse(s));
    }

    sdfg::math::tensor::ReduceNode* node = nullptr;
    if (op_type == "sum") {
        node = static_cast<sdfg::math::tensor::ReduceNode*>(
            &builder.add_library_node<sdfg::math::tensor::SumNode>(block, debug_info, shape_exprs, axes, keepdims)
        );
    } else if (op_type == "max") {
        node = static_cast<sdfg::math::tensor::ReduceNode*>(
            &builder.add_library_node<sdfg::math::tensor::MaxNode>(block, debug_info, shape_exprs, axes, keepdims)
        );
    } else if (op_type == "min") {
        node = static_cast<sdfg::math::tensor::ReduceNode*>(
            &builder.add_library_node<sdfg::math::tensor::MinNode>(block, debug_info, shape_exprs, axes, keepdims)
        );
    } else if (op_type == "mean") {
        node = static_cast<sdfg::math::tensor::ReduceNode*>(
            &builder.add_library_node<sdfg::math::tensor::MeanNode>(block, debug_info, shape_exprs, axes, keepdims)
        );
    } else if (op_type == "std") {
        node = static_cast<sdfg::math::tensor::ReduceNode*>(
            &builder.add_library_node<sdfg::math::tensor::StdNode>(block, debug_info, shape_exprs, axes, keepdims)
        );
    } else if (op_type == "softmax") {
        node = static_cast<sdfg::math::tensor::ReduceNode*>(
            &builder.add_library_node<sdfg::math::tensor::SoftmaxNode>(block, debug_info, shape_exprs, axes, keepdims)
        );
    } else {
        throw std::runtime_error("Unsupported reduce operation: " + op_type);
    }

    auto& in_access = builder.add_access(block, input, debug_info);
    auto& out_access = builder.add_access(block, output, debug_info);

    auto& in_type = builder.subject().type(input);
    auto& out_type = builder.subject().type(output);

    builder.add_computational_memlet(block, in_access, *node, "X", {}, in_type, debug_info);
    builder.add_computational_memlet(block, *node, "Y", out_access, {}, out_type, debug_info);
}
