#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief A pass that eliminates duplicate defines with the same constant value.
 *
 * For example, in the following code, the second definition of `a` inside the loop is redundant and will be removed.
 * a = 5
 * while (cond) {
 *     a = 5  # this define is redundant and will be removed
 *     ...
 * }
 */
class ConstantPropagation : public Pass {
public:
    ConstantPropagation();

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
