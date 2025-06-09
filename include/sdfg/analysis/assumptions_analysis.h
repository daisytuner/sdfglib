#pragma once

#include <set>
#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/assumptions.h"

namespace sdfg {
namespace analysis {

class AssumptionsAnalysis : public Analysis {
   private:
    std::unordered_map<const structured_control_flow::ControlFlowNode*, symbolic::Assumptions>
        assumptions_;

    void traverse(const structured_control_flow::Sequence& root);

    void visit_block(const structured_control_flow::Block* block);

    void visit_sequence(const structured_control_flow::Sequence* sequence);

    void visit_if_else(const structured_control_flow::IfElse* if_else);

    void visit_while(const structured_control_flow::While* while_loop);

    void visit_for(const structured_control_flow::For* for_loop);

    void visit_map(const structured_control_flow::Map* map);

   protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

   public:
    AssumptionsAnalysis(StructuredSDFG& sdfg);

    const symbolic::Assumptions get(const structured_control_flow::ControlFlowNode& node);
};

}  // namespace analysis
}  // namespace sdfg
