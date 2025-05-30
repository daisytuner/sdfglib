#pragma once

#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/kernel.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

class HappensBeforeAnalysis : public Analysis {
    friend class AnalysisManager;

   private:
    structured_control_flow::Sequence& node_;
    std::unordered_map<std::string, std::unordered_map<User*, std::unordered_set<User*>>> results_;

   public:
    HappensBeforeAnalysis(StructuredSDFG& sdfg);

    HappensBeforeAnalysis(StructuredSDFG& sdfg, structured_control_flow::Sequence& node);

    void run(analysis::AnalysisManager& analysis_manager) override;

    /****** Visitor API ******/

    void visit_block(
        analysis::Users& users, structured_control_flow::Block& block,
        std::unordered_set<User*>& open_reads,
        std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write);

    void visit_for(analysis::Users& users, structured_control_flow::For& for_loop,
                   std::unordered_set<User*>& open_reads,
                   std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
                   std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write);

    void visit_if_else(
        analysis::Users& users, structured_control_flow::IfElse& if_loop,
        std::unordered_set<User*>& open_reads,
        std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write);

    void visit_while(
        analysis::Users& users, structured_control_flow::While& while_loop,
        std::unordered_set<User*>& open_reads,
        std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write);

    void visit_return(
        analysis::Users& users, structured_control_flow::Return& return_statement,
        std::unordered_set<User*>& open_reads,
        std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write);

    void visit_kernel(
        analysis::Users& users, structured_control_flow::Kernel& kernel,
        std::unordered_set<User*>& open_reads,
        std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write);

    void visit_sequence(
        analysis::Users& users, structured_control_flow::Sequence& sequence,
        std::unordered_set<User*>& open_reads,
        std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write);

    void visit_map(analysis::Users& users, structured_control_flow::Map& map,
                   std::unordered_set<User*>& open_reads,
                   std::unordered_map<User*, std::unordered_set<User*>>& open_reads_after_writes,
                   std::unordered_map<User*, std::unordered_set<User*>>& closed_reads_after_write);

    /****** Public API ******/

    std::unordered_set<User*> reads_after_write(User& write);

    std::unordered_map<User*, std::unordered_set<User*>> reads_after_writes(
        const std::string& container);

    std::unordered_map<User*, std::unordered_set<User*>> reads_after_write_groups(
        const std::string& container);
};

}  // namespace analysis
}  // namespace sdfg