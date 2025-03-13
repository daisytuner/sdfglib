#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class Kernel : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

   private:
    symbolic::Expression gridDim_x_init_;
    symbolic::Expression gridDim_y_init_;
    symbolic::Expression gridDim_z_init_;

    symbolic::Expression blockDim_x_init_;
    symbolic::Expression blockDim_y_init_;
    symbolic::Expression blockDim_z_init_;

    symbolic::Expression blockIdx_x_init_;
    symbolic::Expression blockIdx_y_init_;
    symbolic::Expression blockIdx_z_init_;

    symbolic::Expression threadIdx_x_init_;
    symbolic::Expression threadIdx_y_init_;
    symbolic::Expression threadIdx_z_init_;

    symbolic::Symbol gridDim_x_;
    symbolic::Symbol gridDim_y_;
    symbolic::Symbol gridDim_z_;

    symbolic::Symbol blockDim_x_;
    symbolic::Symbol blockDim_y_;
    symbolic::Symbol blockDim_z_;

    symbolic::Symbol blockIdx_x_;
    symbolic::Symbol blockIdx_y_;
    symbolic::Symbol blockIdx_z_;

    symbolic::Symbol threadIdx_x_;
    symbolic::Symbol threadIdx_y_;
    symbolic::Symbol threadIdx_z_;

    std::unique_ptr<Sequence> root_;

    std::string suffix_;

    Kernel(size_t element_id, const DebugInfo& debug_info, std::string suffix,
           symbolic::Expression gridDim_x_init = symbolic::symbol("gridDim.x"),
           symbolic::Expression gridDim_y_init = symbolic::symbol("gridDim.y"),
           symbolic::Expression gridDim_z_init = symbolic::symbol("gridDim.z"),
           symbolic::Expression blockDim_x_init = symbolic::symbol("blockDim.x"),
           symbolic::Expression blockDim_y_init = symbolic::symbol("blockDim.y"),
           symbolic::Expression blockDim_z_init = symbolic::symbol("blockDim.z"),
           symbolic::Expression blockIdx_x_init = symbolic::symbol("blockIdx.x"),
           symbolic::Expression blockIdx_y_init = symbolic::symbol("blockIdx.y"),
           symbolic::Expression blockIdx_z_init = symbolic::symbol("blockIdx.z"),
           symbolic::Expression threadIdx_x_init = symbolic::symbol("threadIdx.x"),
           symbolic::Expression threadIdx_y_init = symbolic::symbol("threadIdx.y"),
           symbolic::Expression threadIdx_z_init = symbolic::symbol("threadIdx.z"));

   public:
    Kernel(const Kernel& kernel) = delete;
    Kernel& operator=(const Kernel&) = delete;

    symbolic::Expression gridDim_x_init() const;
    symbolic::Expression gridDim_y_init() const;
    symbolic::Expression gridDim_z_init() const;

    symbolic::Expression blockDim_x_init() const;
    symbolic::Expression blockDim_y_init() const;
    symbolic::Expression blockDim_z_init() const;

    symbolic::Expression blockIdx_x_init() const;
    symbolic::Expression blockIdx_y_init() const;
    symbolic::Expression blockIdx_z_init() const;

    symbolic::Expression threadIdx_x_init() const;
    symbolic::Expression threadIdx_y_init() const;
    symbolic::Expression threadIdx_z_init() const;

    symbolic::Symbol gridDim_x() const;
    symbolic::Symbol gridDim_y() const;
    symbolic::Symbol gridDim_z() const;

    symbolic::Symbol blockDim_x() const;
    symbolic::Symbol blockDim_y() const;
    symbolic::Symbol blockDim_z() const;

    symbolic::Symbol blockIdx_x() const;
    symbolic::Symbol blockIdx_y() const;
    symbolic::Symbol blockIdx_z() const;

    symbolic::Symbol threadIdx_x() const;
    symbolic::Symbol threadIdx_y() const;
    symbolic::Symbol threadIdx_z() const;

    Sequence& root() const;

    std::string suffix() const;

    void replace(const symbolic::Expression& old_expression,
                 const symbolic::Expression& new_expression) override;
};

}  // namespace structured_control_flow
}  // namespace sdfg