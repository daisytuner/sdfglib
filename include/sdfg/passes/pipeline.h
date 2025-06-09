#pragma once

#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/dataflow/dead_reference_elimination.h"
#include "sdfg/passes/dataflow/memlet_propagation.h"
#include "sdfg/passes/dataflow/redundant_array_elimination.h"
#include "sdfg/passes/dataflow/trivial_array_elimination.h"
#include "sdfg/passes/dataflow/view_propagation.h"
#include "sdfg/passes/pass.h"
#include "sdfg/passes/structured_control_flow/block_fusion.h"
#include "sdfg/passes/structured_control_flow/common_assignment_elimination.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/passes/symbolic/condition_propagation.h"
#include "sdfg/passes/symbolic/symbol_propagation.h"

namespace sdfg {
namespace passes {

class Pipeline : public Pass {
   private:
    std::vector<std::unique_ptr<Pass>> passes_;
    std::string name_;

   public:
    Pipeline(const std::string& name);

    virtual std::string name();

    size_t size() const;

    virtual bool run(builder::SDFGBuilder& builder);

    virtual bool run(builder::StructuredSDFGBuilder& builder,
                     analysis::AnalysisManager& analysis_manager);

    template <class T>
    void register_pass() {
        this->passes_.push_back(std::make_unique<T>());
    };

    static Pipeline expression_combine();

    static Pipeline memlet_combine();

    static Pipeline controlflow_simplification();
};

}  // namespace passes
}  // namespace sdfg