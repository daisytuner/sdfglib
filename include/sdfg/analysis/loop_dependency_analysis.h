#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/symbolic/assumptions.h"

namespace sdfg {
namespace analysis {

enum LoopCarriedDependency { RAW, WAW };

typedef std::unordered_map<std::string, LoopCarriedDependency> LoopDependencyAnalysisResult;

class LoopDependencyAnalysis : public Analysis {
   private:
    std::unordered_map<const structured_control_flow::StructuredLoop*, LoopDependencyAnalysisResult>
        results_;

    void analyze(analysis::AnalysisManager& analysis_manager,
                 structured_control_flow::StructuredLoop* loop);

    bool intersects(User* first, User* second, structured_control_flow::StructuredLoop& loop,
                    analysis::UsersView& body_users,
                    analysis::AnalysisManager& analysis_manager) const;

   protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

   public:
    LoopDependencyAnalysis(StructuredSDFG& sdfg);

    LoopDependencyAnalysisResult get(const structured_control_flow::StructuredLoop& loop) const;
};

}  // namespace analysis
}  // namespace sdfg
