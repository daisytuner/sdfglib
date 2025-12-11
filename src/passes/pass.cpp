#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

bool Pass::run(builder::SDFGBuilder& builder, bool create_report) {
    bool applied = this->run_pass(builder);

#ifndef NDEBUG
    builder.subject().validate();
#endif

    return applied;
};

bool Pass::run(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool create_report) {
    bool applied = this->run_pass(builder, analysis_manager);
    this->invalidates(analysis_manager, applied);

#ifndef NDEBUG
    builder.subject().validate();
#endif

    return applied;
};

bool Pass::run_pass(builder::SDFGBuilder& builder) { throw std::logic_error("Not implemented"); };

bool Pass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    throw std::logic_error("Not implemented");
};

void Pass::invalidates(analysis::AnalysisManager& analysis_manager, bool applied) {
    if (applied) {
        analysis_manager.invalidate_all();
    }
};

} // namespace passes
} // namespace sdfg
