#pragma once

#include <set>
#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/analysis.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

class AssumptionsAnalysis : public Analysis {
   private:
    std::vector<std::string> iterators_;
    std::unordered_map<structured_control_flow::ControlFlowNode*, symbolic::Assumptions>
        assumptions_;

    void traverse(structured_control_flow::Sequence& root);

    void visit_block(structured_control_flow::Block* block);

    void visit_sequence(structured_control_flow::Sequence* sequence);

    void visit_if_else(structured_control_flow::IfElse* if_else);

    void visit_while(structured_control_flow::While* while_loop);

    void visit_for(structured_control_flow::For* for_loop);

    void visit_kernel(const structured_control_flow::Kernel* kernel);

    void visit_map(const structured_control_flow::Map* map);

   protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

   public:
    AssumptionsAnalysis(StructuredSDFG& sdfg);

    const symbolic::Assumptions get(structured_control_flow::ControlFlowNode& node);
};

}  // namespace analysis
}  // namespace sdfg
