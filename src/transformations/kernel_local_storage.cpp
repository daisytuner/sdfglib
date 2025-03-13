#include "sdfg/transformations/kernel_local_storage.h"

#include <tuple>
#include <utility>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/kernel.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/utils.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "symengine/integer.h"
#include "symengine/symbol.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace transformations {

KernelLocalStorage::KernelLocalStorage(structured_control_flow::Sequence& parent,
                                       structured_control_flow::For& outer_loop,
                                       structured_control_flow::For& inner_loop,
                                       std::string container)
    : parent_(parent), outer_loop_(outer_loop), inner_loop_(inner_loop), container_(container){};

std::string KernelLocalStorage::name() { return "KernelLocalStorage"; };

bool KernelLocalStorage::reads_container(std::string container, const Sequence& sequence,
                                         analysis::UsersView& body_users) {
    for (auto& user : body_users.reads(container)) {
        auto& subsets = user->subsets();
        for (auto& subset : subsets) {
            return true;
        }
    }
    return true;
}

bool KernelLocalStorage::uses_inner_indvar(const structured_control_flow::Kernel* kernel,
                                           const structured_control_flow::Sequence& body,
                                           analysis::UsersView& body_users) {
    bool result = false;
    for (auto& user : body_users.reads(this->container_)) {
        auto& subsets = user->subsets();
        if (subsets.size() == 1) {            // TODO: Handle multiple subsets
            if (subsets.at(0).size() == 1) {  // TODO: Handle multiple dimensions
                result |= symbolic::uses(subsets.at(0).at(0), inner_loop_.indvar());
            }
        }
    }
    return result;
};

std::tuple<symbolic::Integer, symbolic::Integer, symbolic::Integer> KernelLocalStorage::dim_size(
    const structured_control_flow::Kernel* kernel, symbolic::Assumptions& assumptions) {
    symbolic::Integer x_dim_size = SymEngine::null;
    symbolic::Integer y_dim_size = SymEngine::null;
    symbolic::Integer z_dim_size = SymEngine::null;

    auto x_ub = assumptions[kernel->blockDim_x()].upper_bound();
    x_dim_size = SymEngine::rcp_static_cast<const SymEngine::Integer>(x_ub);

    auto y_ub = assumptions[kernel->blockDim_y()].upper_bound();
    y_dim_size = SymEngine::rcp_static_cast<const SymEngine::Integer>(y_ub);

    auto z_ub = assumptions[kernel->blockDim_z()].upper_bound();
    z_dim_size = SymEngine::rcp_static_cast<const SymEngine::Integer>(z_ub);

    return std::make_tuple(x_dim_size, y_dim_size, z_dim_size);
};

bool KernelLocalStorage::can_be_applied(Schedule& schedule) {
    auto& analysis_manager = schedule.analysis_manager();
    auto& builder = schedule.builder();

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    auto& inner_body = this->inner_loop_.root();

    // Criterion: Check if parent is a kernel
    if (root.size() != 1) {
        return false;
    }
    auto kernel = dynamic_cast<const sdfg::structured_control_flow::Kernel*>(&root.at(0).first);
    if (!kernel) {
        return false;
    }

    // Criterion: Container is pointer to scalar type
    auto& type = sdfg.type(this->container_);
    auto pointer_type = dynamic_cast<const types::Pointer*>(&type);
    if (!pointer_type) {
        return false;
    }
    if (!dynamic_cast<const types::Scalar*>(&pointer_type->pointee_type())) {
        return false;
    }

    // Criterion: Iteration count is known and an Integer
    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto assumptions = assumptions_analysis.get(inner_body);
    symbolic::Integer iteration_count = get_iteration_count(inner_loop_);
    if (iteration_count == SymEngine::null) {
        return false;
    }

    // Criterion: All block dimensions are known and an Integer
    auto x_ub = assumptions[kernel->blockDim_x()].upper_bound();
    auto x_lb = assumptions[kernel->blockDim_x()].lower_bound();
    if (!symbolic::eq(x_ub, x_lb)) {
        return false;
    }
    if (!SymEngine::is_a<SymEngine::Integer>(*x_ub)) {
        return false;
    }

    auto y_ub = assumptions[kernel->blockDim_y()].upper_bound();
    auto y_lb = assumptions[kernel->blockDim_y()].lower_bound();
    if (!symbolic::eq(y_ub, y_lb)) {
        return false;
    }
    if (!SymEngine::is_a<SymEngine::Integer>(*y_ub)) {
        return false;
    }

    auto z_ub = assumptions[kernel->blockDim_z()].upper_bound();
    auto z_lb = assumptions[kernel->blockDim_z()].lower_bound();
    if (!symbolic::eq(z_ub, z_lb)) {
        return false;
    }
    if (!SymEngine::is_a<SymEngine::Integer>(*z_ub)) {
        return false;
    }

    // Criteria related to memory accesses
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView inner_body_users(users, inner_body);

    // Criterion: Container is read-only
    if (!inner_body_users.writes(this->container_).empty() ||
        !inner_body_users.views(this->container_).empty() ||
        !inner_body_users.moves(this->container_).empty()) {
        return false;
    }
    if (inner_body_users.reads(this->container_).empty()) {
        return false;
    }

    // Collect moving symbols

    // Criterion: Memory accesses do not depend on moving symbols
    for (auto& user : inner_body_users.uses(this->container_)) {
        auto& subsets = user->subsets();
        for (auto& subset : subsets) {
            for (auto& expr : subset) {
                for (auto& atom : symbolic::atoms(expr)) {
                    if (SymEngine::is_a<SymEngine::Symbol>(*atom)) {
                        auto symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                        if (!inner_body_users.moves(symbol->get_name()).empty()) {
                            return false;
                        }
                    }
                }
            }
        }
    }

    // Criterion: Check if all memory accesses are affine w.r.t the inner loop index

    // Limitations: single memory access
    if (inner_body_users.reads(this->container_).size() != 1) {
        return false;
    }
    auto read = inner_body_users.reads(this->container_).at(0);
    if (read->subsets().size() != 1) {
        return false;
    }
    auto subset = read->subsets().at(0);
    if (subset.size() != 1) {
        return false;
    }

    // Criterion: Memory access is polynomial of
    // c_0 * a + c_1 * b + c_2 * c + c_3 * k, where a, b, c are x-threads, y-threads, z-threads
    // and k is the inner loop index
    auto a = symbolic::add(kernel->threadIdx_x(),
                           symbolic::mul(kernel->blockIdx_x(), kernel->blockDim_x()));
    auto b = symbolic::add(kernel->threadIdx_y(),
                           symbolic::mul(kernel->blockIdx_y(), kernel->blockDim_y()));
    auto c = symbolic::add(kernel->threadIdx_z(),
                           symbolic::mul(kernel->blockIdx_z(), kernel->blockDim_z()));

    auto access = subset.at(0);
    access = symbolic::subs(access, a, symbolic::symbol("a"));
    access = symbolic::subs(access, b, symbolic::symbol("b"));
    access = symbolic::subs(access, c, symbolic::symbol("c"));

    // TODO: Real structuring of polynomial
    /* auto poly = symbolic::polynomial(access);
    if (poly == SymEngine::null) {
        return false;
    } */

    return true;
};

void KernelLocalStorage::apply(Schedule& schedule) {
    auto& analysis_manager = schedule.analysis_manager();
    auto& builder = schedule.builder();
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    auto& users = analysis_manager.get<analysis::Users>();

    auto& inner_body = this->inner_loop_.root();
    analysis::UsersView inner_body_users(users, inner_body);

    auto& assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto assumptions = assumptions_analysis.get(inner_body);

    const sdfg::structured_control_flow::Kernel* kernel =
        dynamic_cast<const sdfg::structured_control_flow::Kernel*>(
            &schedule.sdfg().root().at(0).first);

    symbolic::Integer iteration_count = get_iteration_count(inner_loop_);

    auto [x_dim_size, y_dim_size, z_dim_size] = dim_size(kernel, assumptions);

    // calculate shared memory shape
    std::tuple<symbolic::Integer, symbolic::Integer, symbolic::Integer, symbolic::Integer>
        shared_memory_shape = std::make_tuple(iteration_count, x_dim_size, y_dim_size, z_dim_size);

    // Get primitive type of container
    const types::Pointer* pointer =
        static_cast<const types::Pointer*>(&sdfg.type(this->container_));
    const types::Scalar* base_type =
        static_cast<const types::Scalar*>(&pointer->pointee_type());  // must be scalar or struct

    const types::Scalar type(base_type->primitive_type(), types::DeviceLocation::nvptx, 3);

    // Allocate shared memory before the outer loop, starting from z, y, x, iteration_count
    types::Array shared_memory(type, std::get<0>(shared_memory_shape), types::DeviceLocation::nvptx,
                               3);
    types::Array shared_memory_x(shared_memory, std::get<1>(shared_memory_shape),
                                 types::DeviceLocation::nvptx, 3);
    types::Array shared_memory_y(shared_memory_x, std::get<2>(shared_memory_shape),
                                 types::DeviceLocation::nvptx, 3);
    types::Array shared_memory_z(shared_memory_y, std::get<3>(shared_memory_shape),
                                 types::DeviceLocation::nvptx, 3);

    builder.add_container("__daisy_share_" + this->container_, shared_memory_z);

    bool has_tid_x = false;
    bool has_tid_y = false;
    bool has_tid_z = false;
    for (auto container : sdfg.containers()) {
        if (container == kernel->threadIdx_x()->get_name()) {
            has_tid_x = true;
        }
        if (container == kernel->threadIdx_y()->get_name()) {
            has_tid_y = true;
        }
        if (container == kernel->threadIdx_z()->get_name()) {
            has_tid_z = true;
        }
    }
    if (!has_tid_x) {
        builder.add_container(kernel->threadIdx_x()->get_name(),
                              types::Scalar(types::PrimitiveType::Int32));
    }
    if (!has_tid_y) {
        builder.add_container(kernel->threadIdx_y()->get_name(),
                              types::Scalar(types::PrimitiveType::Int32));
    }
    if (!has_tid_z) {
        builder.add_container(kernel->threadIdx_z()->get_name(),
                              types::Scalar(types::PrimitiveType::Int32));
    }

    // Deconstrunct array accesses into dimensions
    // Read from global memory to shared memory. Ensure the data access bounds are correct
    auto& outer_body = this->outer_loop_.root();

    builder.add_container("__daisy_shared_indvar_" + this->container_,
                          types::Scalar(types::Scalar(types::PrimitiveType::Int32)));

    symbolic::Symbol indvar = symbolic::symbol("__daisy_shared_indvar_" + this->container_);
    symbolic::Expression init_expr =
        symbolic::subs(inner_loop_.init(), inner_loop_.indvar(), indvar);
    symbolic::Condition condition_expr =
        symbolic::subs(inner_loop_.condition(), inner_loop_.indvar(), indvar);
    symbolic::Expression update_expr =
        symbolic::subs(inner_loop_.update(), inner_loop_.indvar(), indvar);
    auto& copyin_for = builder
                           .add_for_before(outer_body, this->inner_loop_, indvar, condition_expr,
                                           init_expr, update_expr)
                           .first;

    auto& copyin_block = builder.add_block(copyin_for.root());

    auto& access_node_in = builder.add_access(copyin_block, this->container_);
    auto& access_node_out = builder.add_access(copyin_block, "__daisy_share_" + this->container_);
    auto& tasklet_copy_in = builder.add_tasklet(copyin_block, data_flow::TaskletCode::assign,
                                                {"_out", *base_type}, {{"_in", *base_type}});

    symbolic::Expression read_expr =
        inner_body_users.reads(this->container_).at(0)->subsets().at(0).at(0);
    read_expr = symbolic::subs(read_expr, inner_loop_.indvar(), indvar);
    builder.add_memlet(copyin_block, access_node_in, "void", tasklet_copy_in, "_in", {read_expr});

    // Set the access indices

    std::tuple<symbolic::Expression, symbolic::Expression, symbolic::Expression,
               symbolic::Expression>
        shared_access_scheme_write =
            std::make_tuple(kernel->threadIdx_z(), kernel->threadIdx_y(), kernel->threadIdx_x(),
                            symbolic::sub(indvar, outer_loop_.indvar()));
    builder.add_memlet(
        copyin_block, tasklet_copy_in, "_out", access_node_out, "void",
        {std::get<0>(shared_access_scheme_write), std::get<1>(shared_access_scheme_write),
         std::get<2>(shared_access_scheme_write), std::get<3>(shared_access_scheme_write)});

    // Replace global memory accesses with shared memory accesses
    builder.add_container("__daisy_share_wrapper_" + this->container_, *base_type);
    inner_body.replace(symbolic::symbol(this->container_),
                       symbolic::symbol("__daisy_share_wrapper_" + this->container_));

    auto& read_block =
        builder.add_block_before(inner_loop_.root(), inner_loop_.root().at(0).first).first;
    auto& read_node_in = builder.add_access(read_block, "__daisy_share_" + this->container_);
    auto& read_node_out =
        builder.add_access(read_block, "__daisy_share_wrapper_" + this->container_);

    auto& tasklet_read = builder.add_tasklet(read_block, data_flow::TaskletCode::assign,
                                             {"_out", *base_type}, {{"_in", *base_type}});

    std::tuple<symbolic::Expression, symbolic::Expression, symbolic::Expression,
               symbolic::Expression>
        shared_access_scheme_read =
            std::make_tuple(kernel->threadIdx_z(), kernel->threadIdx_y(), kernel->threadIdx_x(),
                            symbolic::sub(inner_loop_.indvar(), outer_loop_.indvar()));

    builder.add_memlet(
        read_block, read_node_in, "void", tasklet_read, "_in",
        {std::get<0>(shared_access_scheme_read), std::get<1>(shared_access_scheme_read),
         std::get<2>(shared_access_scheme_read), std::get<3>(shared_access_scheme_read)});

    builder.add_memlet(read_block, tasklet_read, "_out", read_node_out, "void", {});

    auto& sync_block = builder.add_block_before(outer_body, this->inner_loop_).first;
    builder.add_library_node(sync_block, data_flow::LibraryNodeType::LocalBarrier, {}, {}, true);

    // End of transformation

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
