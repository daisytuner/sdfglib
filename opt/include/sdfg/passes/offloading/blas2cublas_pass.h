#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace cuda {


/**
 * Name is wrong, as it is no longer MathNode specific.
 */
class Blas2CuBlas : public visitor::StructuredSDFGVisitor {
public:
    Blas2CuBlas(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "Blas2CuBlasPass"; };
    bool accept(structured_control_flow::Block& node) override;
};

typedef passes::VisitorPass<Blas2CuBlas> Blas2CuBlasPass;

} // namespace cuda
} // namespace sdfg
