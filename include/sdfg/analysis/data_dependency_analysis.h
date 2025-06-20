#pragma once

#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

class DataDependencyAnalysis : public Analysis {
    friend class AnalysisManager;

   private:
    structured_control_flow::Sequence& node_;
    std::unordered_map<std::string, std::unordered_map<User*, std::unordered_set<User*>>> results_;

    bool overwrites(User& previous, User& current,
                    analysis::AssumptionsAnalysis& assumptions_analysis);

    bool reads(User& previous, User& current, analysis::AssumptionsAnalysis& assumptions_analysis);

   public:
    DataDependencyAnalysis(StructuredSDFG& sdfg);

    DataDependencyAnalysis(StructuredSDFG& sdfg, structured_control_flow::Sequence& node);

    void run(analysis::AnalysisManager& analysis_manager) override;

    /****** Visitor API ******/

    void visit_block(analysis::Users& users, analysis::AssumptionsAnalysis& assumptions_analysis,
                     structured_control_flow::Block& block, std::unordered_set<User*>& undefined,
                     std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
                     std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions);

    void visit_for(analysis::Users& users, analysis::AssumptionsAnalysis& assumptions_analysis,
                   structured_control_flow::For& for_loop, std::unordered_set<User*>& undefined,
                   std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
                   std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions);

    void visit_if_else(analysis::Users& users, analysis::AssumptionsAnalysis& assumptions_analysis,
                       structured_control_flow::IfElse& if_loop,
                       std::unordered_set<User*>& undefined,
                       std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
                       std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions);

    void visit_while(analysis::Users& users, analysis::AssumptionsAnalysis& assumptions_analysis,
                     structured_control_flow::While& while_loop,
                     std::unordered_set<User*>& undefined,
                     std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
                     std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions);

    void visit_return(analysis::Users& users, analysis::AssumptionsAnalysis& assumptions_analysis,
                      structured_control_flow::Return& return_statement,
                      std::unordered_set<User*>& undefined,
                      std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
                      std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions);

    void visit_sequence(analysis::Users& users, analysis::AssumptionsAnalysis& assumptions_analysis,
                        structured_control_flow::Sequence& sequence,
                        std::unordered_set<User*>& undefined,
                        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
                        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions);

    void visit_map(analysis::Users& users, analysis::AssumptionsAnalysis& assumptions_analysis,
                   structured_control_flow::Map& map, std::unordered_set<User*>& undefined,
                   std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
                   std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions);

    /****** Public API ******/

    /**
     * @brief Get the users (reads) of a definition (write).
     *
     * @param write the definition (write)
     * @return The users (reads) of the definition.
     */
    std::unordered_set<User*> defines(User& write);

    /**
     * @brief Get all definitions (writes) and their users (reads).
     *
     * @param The container of the definitions.
     * @return The definitions and their users.
     */
    std::unordered_map<User*, std::unordered_set<User*>> definitions(const std::string& container);

    /**
     * @brief Get all definitions (writes) for each user (reads).
     *
     * @param The container of the definitions.
     * @return The users (reads) and their definitions (writes).
     */
    std::unordered_map<User*, std::unordered_set<User*>> defined_by(const std::string& container);

    /**
     * @brief Get all definitions (writes) for a user (read).
     *
     * @param The user (read).
     * @return The definitions (writes) of the user.
     */
    std::unordered_set<User*> defined_by(User& read);
};

}  // namespace analysis
}  // namespace sdfg