#include "sdfg/transformations/loop_to_kernel_dim.h"

#include <utility>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_parallelism_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/kernel.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/utils.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace transformations {

LoopToKernelDim::LoopToKernelDim(structured_control_flow::Sequence& parent,
                                 structured_control_flow::For& loop)
    : parent_(parent), loop_(loop) {};

std::string LoopToKernelDim::name() { return "LoopToKernelDim"; };

bool LoopToKernelDim::can_be_applied(Schedule& schedule) {
    auto& analysis_manager = schedule.analysis_manager();
    sdfg::passes::Pipeline expression_combine = sdfg::passes::Pipeline::expression_combine();
    sdfg::passes::Pipeline memlet_combine = sdfg::passes::Pipeline::memlet_combine();
    memlet_combine.run(schedule.builder(), analysis_manager);
    auto& builder = schedule.builder();

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Criterion: Check if ancestor is a kernel
    if (root.size() != 1) {
        return false;
    }
    auto kernel = dynamic_cast<const sdfg::structured_control_flow::Kernel*>(&root.at(0).first);
    if (!kernel) {
        return false;
    }

    // Criterion: Iteration count is known and an Integer
    symbolic::Integer iteration_count = get_iteration_count(loop_);
    if (iteration_count == SymEngine::null) {
        return false;
    }

    // Criterion: Indvar is only used to access arrays
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, loop_.root());
    for (auto user : body_users.reads(loop_.indvar()->get_name())) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element())) {
            return false;
        }
    }

    // Criterion: Kernel dimensions are known and an Integer
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto assumptions = assumptions_analysis.get(loop_.root());
    auto x_dim_size = assumptions[kernel->blockDim_x()].integer_value();
    if (x_dim_size == SymEngine::null) {
        return false;
    }
    auto y_dim_size = assumptions[kernel->blockDim_y()].integer_value();
    if (y_dim_size == SymEngine::null) {
        return false;
    }
    auto z_dim_size = assumptions[kernel->blockDim_z()].integer_value();
    if (z_dim_size == SymEngine::null) {
        return false;
    }

    // Criterion: Available kernel dimensions is free
    bool x_dim_available = false;
    bool y_dim_available = false;
    bool z_dim_available = false;

    if (!symbolic::eq(x_dim_size, symbolic::integer(1))) {
        if (body_users.reads(kernel->threadIdx_x()->get_name()).empty()) {
            x_dim_available = true;
        }
    }
    if (!symbolic::eq(y_dim_size, symbolic::integer(1))) {
        if (body_users.reads(kernel->threadIdx_y()->get_name()).empty()) {
            y_dim_available = true;
        }
    }
    if (!symbolic::eq(z_dim_size, symbolic::integer(1))) {
        if (body_users.reads(kernel->threadIdx_z()->get_name()).empty()) {
            z_dim_available = true;
        }
    }
    if (!x_dim_available && !y_dim_available && !z_dim_available) {
        return false;
    }

    // Criterion: Unused kernel dimension is bigger or equal to loop iteration count
    bool x_match = false;
    bool y_match = false;
    bool z_match = false;
    if (x_dim_available) {
        auto cond = symbolic::Ge(x_dim_size, iteration_count);
        if (symbolic::is_true(cond)) {
            x_match = true;
        }
    }
    if (y_dim_available) {
        auto cond = symbolic::Ge(y_dim_size, iteration_count);
        if (symbolic::is_true(cond)) {
            y_match = true;
        }
    }
    if (z_dim_available) {
        auto cond = symbolic::Ge(z_dim_size, iteration_count);
        if (symbolic::is_true(cond)) {
            z_match = true;
        }
    }
    if (!x_match && !y_match && !z_match) {
        return false;
    }

    // Criterion: Loop is a map
    auto& analysis = analysis_manager.get<analysis::DataParallelismAnalysis>();
    auto& data_dependencies = analysis.get(this->loop_);
    for (auto& dep : data_dependencies) {
        auto& container = dep.first;
        auto& dep_type = dep.second;
        if (dep_type < analysis::Parallelism::PARALLEL) {
            return false;
        }
    }

    return true;
};

void LoopToKernelDim::apply(Schedule& schedule) {
    auto& analysis_manager = schedule.analysis_manager();
    auto& builder = schedule.builder();
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    auto kernel = static_cast<const sdfg::structured_control_flow::Kernel*>(&root.at(0).first);

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto assumptions = assumptions_analysis.get(loop_.root());
    auto x_dim_size = assumptions[kernel->blockDim_x()].integer_value();
    auto y_dim_size = assumptions[kernel->blockDim_y()].integer_value();
    auto z_dim_size = assumptions[kernel->blockDim_z()].integer_value();

    bool x_dim_available = false;
    bool y_dim_available = false;
    bool z_dim_available = false;
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, loop_.root());

    if (!symbolic::eq(x_dim_size, symbolic::integer(1))) {
        if (body_users.reads(kernel->threadIdx_x()->get_name()).empty()) {
            x_dim_available = true;
        }
    }
    if (!symbolic::eq(y_dim_size, symbolic::integer(1))) {
        if (body_users.reads(kernel->threadIdx_y()->get_name()).empty()) {
            y_dim_available = true;
        }
    }
    if (!symbolic::eq(z_dim_size, symbolic::integer(1))) {
        if (body_users.reads(kernel->threadIdx_z()->get_name()).empty()) {
            z_dim_available = true;
        }
    }

    bool x_match = false;
    bool y_match = false;
    bool z_match = false;
    symbolic::Integer iteration_count = get_iteration_count(loop_);
    if (x_dim_available) {
        auto cond = symbolic::Ge(x_dim_size, iteration_count);
        if (symbolic::is_true(cond)) {
            x_match = true;
        }
    }
    if (y_dim_available) {
        auto cond = symbolic::Ge(y_dim_size, iteration_count);
        if (symbolic::is_true(cond)) {
            y_match = true;
        }
    }
    if (z_dim_available) {
        auto cond = symbolic::Ge(z_dim_size, iteration_count);
        if (symbolic::is_true(cond)) {
            z_match = true;
        }
    }

    auto target_dim = kernel->threadIdx_z();
    if (x_match) {
        target_dim = kernel->threadIdx_x();
    } else if (y_match) {
        target_dim = kernel->threadIdx_y();
    }

    auto& parent = builder.parent(loop_);
    auto& if_else = builder.add_if_else_before(parent, loop_).first;
    auto condition =
        symbolic::subs(loop_.condition(), loop_.indvar(), symbolic::add(target_dim, loop_.init()));
    auto& branch = builder.add_case(if_else, condition);

    builder.insert_children(branch, loop_.root(), 0);
    branch.replace(loop_.indvar(), symbolic::add(target_dim, loop_.init()));
    builder.remove_child(parent, loop_);

    analysis_manager.invalidate_all();
    passes::SequenceFusion sf_pass;
    passes::DeadCFGElimination dce_pass;
    bool applies = false;
    do {
        applies = false;
        applies |= dce_pass.run(schedule.builder(), analysis_manager);
        applies |= sf_pass.run(schedule.builder(), analysis_manager);
    } while (applies);
};

}  // namespace transformations
}  // namespace sdfg
