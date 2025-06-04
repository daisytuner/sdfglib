#include "sdfg/structured_control_flow/kernel.h"

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace structured_control_flow {

Kernel::Kernel(const DebugInfo& debug_info, std::string suffix, symbolic::Expression gridDim_x_init,
               symbolic::Expression gridDim_y_init, symbolic::Expression gridDim_z_init,
               symbolic::Expression blockDim_x_init, symbolic::Expression blockDim_y_init,
               symbolic::Expression blockDim_z_init, symbolic::Expression blockIdx_x_init,
               symbolic::Expression blockIdx_y_init, symbolic::Expression blockIdx_z_init,
               symbolic::Expression threadIdx_x_init, symbolic::Expression threadIdx_y_init,
               symbolic::Expression threadIdx_z_init)
    : ControlFlowNode(debug_info),
      gridDim_x_init_(gridDim_x_init),
      gridDim_y_init_(gridDim_y_init),
      gridDim_z_init_(gridDim_z_init),
      blockDim_x_init_(blockDim_x_init),
      blockDim_y_init_(blockDim_y_init),
      blockDim_z_init_(blockDim_z_init),
      blockIdx_x_init_(blockIdx_x_init),
      blockIdx_y_init_(blockIdx_y_init),
      blockIdx_z_init_(blockIdx_z_init),
      threadIdx_x_init_(threadIdx_x_init),
      threadIdx_y_init_(threadIdx_y_init),
      threadIdx_z_init_(threadIdx_z_init),
      suffix_(suffix) {
    this->root_ = std::unique_ptr<Sequence>(new Sequence(debug_info));

    this->gridDim_x_ = symbolic::symbol("__daisy_gridDim_x_" + suffix);
    this->gridDim_y_ = symbolic::symbol("__daisy_gridDim_y_" + suffix);
    this->gridDim_z_ = symbolic::symbol("__daisy_gridDim_z_" + suffix);

    this->blockDim_x_ = symbolic::symbol("__daisy_blockDim_x_" + suffix);
    this->blockDim_y_ = symbolic::symbol("__daisy_blockDim_y_" + suffix);
    this->blockDim_z_ = symbolic::symbol("__daisy_blockDim_z_" + suffix);

    this->blockIdx_x_ = symbolic::symbol("__daisy_blockIdx_x_" + suffix);
    this->blockIdx_y_ = symbolic::symbol("__daisy_blockIdx_y_" + suffix);
    this->blockIdx_z_ = symbolic::symbol("__daisy_blockIdx_z_" + suffix);

    this->threadIdx_x_ = symbolic::symbol("__daisy_threadIdx_x_" + suffix);
    this->threadIdx_y_ = symbolic::symbol("__daisy_threadIdx_y_" + suffix);
    this->threadIdx_z_ = symbolic::symbol("__daisy_threadIdx_z_" + suffix);
};

symbolic::Expression Kernel::gridDim_x_init() const { return this->gridDim_x_init_; }

symbolic::Expression Kernel::gridDim_y_init() const { return this->gridDim_y_init_; }

symbolic::Expression Kernel::gridDim_z_init() const { return this->gridDim_z_init_; }

symbolic::Expression Kernel::blockDim_x_init() const { return this->blockDim_x_init_; }

symbolic::Expression Kernel::blockDim_y_init() const { return this->blockDim_y_init_; }

symbolic::Expression Kernel::blockDim_z_init() const { return this->blockDim_z_init_; }

symbolic::Expression Kernel::blockIdx_x_init() const { return this->blockIdx_x_init_; }

symbolic::Expression Kernel::blockIdx_y_init() const { return this->blockIdx_y_init_; }

symbolic::Expression Kernel::blockIdx_z_init() const { return this->blockIdx_z_init_; }

symbolic::Expression Kernel::threadIdx_x_init() const { return this->threadIdx_x_init_; }

symbolic::Expression Kernel::threadIdx_y_init() const { return this->threadIdx_y_init_; }

symbolic::Expression Kernel::threadIdx_z_init() const { return this->threadIdx_z_init_; }

symbolic::Symbol Kernel::gridDim_x() const { return this->gridDim_x_; }

symbolic::Symbol Kernel::gridDim_y() const { return this->gridDim_y_; }

symbolic::Symbol Kernel::gridDim_z() const { return this->gridDim_z_; }

symbolic::Symbol Kernel::blockDim_x() const { return this->blockDim_x_; }

symbolic::Symbol Kernel::blockDim_y() const { return this->blockDim_y_; }

symbolic::Symbol Kernel::blockDim_z() const { return this->blockDim_z_; }

symbolic::Symbol Kernel::blockIdx_x() const { return this->blockIdx_x_; }

symbolic::Symbol Kernel::blockIdx_y() const { return this->blockIdx_y_; }

symbolic::Symbol Kernel::blockIdx_z() const { return this->blockIdx_z_; }

symbolic::Symbol Kernel::threadIdx_x() const { return this->threadIdx_x_; }

symbolic::Symbol Kernel::threadIdx_y() const { return this->threadIdx_y_; }

symbolic::Symbol Kernel::threadIdx_z() const { return this->threadIdx_z_; }

Sequence& Kernel::root() const { return *this->root_; };

std::string Kernel::suffix() const { return this->suffix_; }

void Kernel::replace(const symbolic::Expression& old_expression,
                     const symbolic::Expression& new_expression) {
    if (symbolic::eq(this->gridDim_x_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->gridDim_x_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    if (symbolic::eq(this->gridDim_y_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->gridDim_y_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    if (symbolic::eq(this->gridDim_z_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->gridDim_z_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    if (symbolic::eq(this->blockDim_x_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->blockDim_x_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    if (symbolic::eq(this->blockDim_y_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->blockDim_y_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    if (symbolic::eq(this->blockDim_z_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->blockDim_z_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    if (symbolic::eq(this->blockIdx_x_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->blockIdx_x_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    if (symbolic::eq(this->blockIdx_y_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->blockIdx_y_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    if (symbolic::eq(this->blockIdx_z_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->blockIdx_z_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    if (symbolic::eq(this->threadIdx_x_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->threadIdx_x_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    if (symbolic::eq(this->threadIdx_y_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->threadIdx_y_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    if (symbolic::eq(this->threadIdx_z_, old_expression)) {
        assert(SymEngine::is_a<SymEngine::Symbol>(*new_expression));
        this->threadIdx_z_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }

    this->blockIdx_x_init_ = symbolic::subs(this->blockIdx_x_init_, old_expression, new_expression);
    this->blockIdx_y_init_ = symbolic::subs(this->blockIdx_y_init_, old_expression, new_expression);
    this->blockIdx_z_init_ = symbolic::subs(this->blockIdx_z_init_, old_expression, new_expression);
    this->threadIdx_x_init_ =
        symbolic::subs(this->threadIdx_x_init_, old_expression, new_expression);
    this->threadIdx_y_init_ =
        symbolic::subs(this->threadIdx_y_init_, old_expression, new_expression);
    this->threadIdx_z_init_ =
        symbolic::subs(this->threadIdx_z_init_, old_expression, new_expression);
    this->blockDim_x_init_ = symbolic::subs(this->blockDim_x_init_, old_expression, new_expression);
    this->blockDim_y_init_ = symbolic::subs(this->blockDim_y_init_, old_expression, new_expression);
    this->blockDim_z_init_ = symbolic::subs(this->blockDim_z_init_, old_expression, new_expression);
    this->gridDim_x_init_ = symbolic::subs(this->gridDim_x_init_, old_expression, new_expression);
    this->gridDim_y_init_ = symbolic::subs(this->gridDim_y_init_, old_expression, new_expression);
    this->gridDim_z_init_ = symbolic::subs(this->gridDim_z_init_, old_expression, new_expression);

    this->root_->replace(old_expression, new_expression);
};

}  // namespace structured_control_flow
}  // namespace sdfg
