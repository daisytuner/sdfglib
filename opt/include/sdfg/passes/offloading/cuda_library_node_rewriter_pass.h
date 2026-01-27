#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace cuda {

class CudaLibraryNodeRewriter : public visitor::StructuredSDFGVisitor {
public:
    CudaLibraryNodeRewriter(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "CudaLibraryNodeRewriterPass"; };
    bool accept(structured_control_flow::Block& node) override;
};

typedef passes::VisitorPass<CudaLibraryNodeRewriter> CudaLibraryNodeRewriterPass;

} // namespace cuda
} // namespace sdfg
