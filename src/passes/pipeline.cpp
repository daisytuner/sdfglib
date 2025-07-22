#include "sdfg/passes/pipeline.h"

namespace sdfg {
namespace passes {

Pipeline::Pipeline(const std::string& name)
    : Pass(), name_(name) {

      };

std::string Pipeline::name() { return this->name_; };

size_t Pipeline::size() const { return this->passes_.size(); };

bool Pipeline::run(builder::SDFGBuilder& builder) {
    bool applied = false;

    bool applied_pipeline;
    do {
        applied_pipeline = false;
        for (auto& pass : this->passes_) {
            bool applied_pass = false;
            do {
                applied_pass = pass->run(builder);
                applied_pipeline |= applied_pass;
            } while (applied_pass);
        }
        applied |= applied_pipeline;
    } while (applied_pipeline);

    return applied;
};

bool Pipeline::run(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;
    bool applied_pipeline;
    do {
        applied_pipeline = false;
        for (auto& pass : this->passes_) {
            bool applied_pass = false;
            do {
                applied_pass = pass->run(builder, analysis_manager);
                applied_pipeline |= applied_pass;
            } while (applied_pass);
        }
        applied |= applied_pipeline;
    } while (applied_pipeline);

    return applied;
};

Pipeline Pipeline::expression_combine() {
    Pipeline p("ExpressionCombine");

    p.register_pass<SymbolPropagation>();
    p.register_pass<ConstantElimination>();
    p.register_pass<DeadDataElimination>();
    p.register_pass<SymbolEvolution>();

    return p;
};

Pipeline Pipeline::memlet_combine() {
    Pipeline p("MemletCombine");

    p.register_pass<ReferencePropagation>();
    p.register_pass<DeadReferenceElimination>();

    return p;
};

Pipeline Pipeline::controlflow_simplification() {
    Pipeline p("ControlFlowSimplification");

    p.register_pass<DeadCFGElimination>();
    p.register_pass<BlockFusionPass>();
    p.register_pass<SequenceFusion>();

    return p;
};

Pipeline Pipeline::data_parallelism() {
    Pipeline p("DataParallelism");

    p.register_pass<For2MapPass>();
    p.register_pass<SymbolPropagation>();
    p.register_pass<DeadDataElimination>();

    return p;
};

} // namespace passes
} // namespace sdfg
