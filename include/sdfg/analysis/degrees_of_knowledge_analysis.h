#pragma once

#include <utility>
#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

enum class DegreesOfKnowledgeClassification { Scalar, Bound, Unbound };

class DegreesOfKnowledgeAnalysis : public Analysis {
private:
    symbolic::SymbolSet while_symbols_;

    std::unordered_map<const structured_control_flow::Map*, std::pair<symbolic::Expression, symbolic::Expression>>
        number_of_maps_;
    std::unordered_map<const structured_control_flow::Map*, std::pair<symbolic::Expression, symbolic::SymbolSet>>
        size_of_a_map_;
    std::unordered_map<const structured_control_flow::Map*, symbolic::Expression> load_of_a_map_;
    std::unordered_map<const structured_control_flow::Map*, symbolic::Expression> balance_of_a_map_;

    void number_analysis(
        AnalysisManager& analysis_manager,
        symbolic::Expression base_iterations,
        bool branched,
        structured_control_flow::ControlFlowNode* node
    );
    void size_analysis(AnalysisManager& analysis_manager);
    void load_analysis();
    void balance_analysis();

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    DegreesOfKnowledgeAnalysis(StructuredSDFG& sdfg);

    std::pair<symbolic::Expression, DegreesOfKnowledgeClassification> number_of_maps(const structured_control_flow::Map*
                                                                                         node) const;

    std::pair<symbolic::Expression, DegreesOfKnowledgeClassification> size_of_a_map(const structured_control_flow::Map*
                                                                                        node) const;

    std::pair<symbolic::Expression, DegreesOfKnowledgeClassification> load_of_a_map(const structured_control_flow::Map*
                                                                                        node) const;

    std::pair<symbolic::Expression, DegreesOfKnowledgeClassification> balance_of_a_map(const structured_control_flow::Map*
                                                                                           node) const;
};

} // namespace analysis
} // namespace sdfg
