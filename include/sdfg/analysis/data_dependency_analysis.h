#pragma once

#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace analysis {

enum LoopCarriedDependency {
    LOOP_CARRIED_DEPENDENCY_READ_WRITE,
    LOOP_CARRIED_DEPENDENCY_WRITE_WRITE,
};

/**
 * @brief Analysis to compute data dependencies between definitions (writes) and uses (reads) of containers.
 *
 * A definition is a write to a container.
 * A use is a read from that container after the definition.
 * A definition is closed when a new definition dominates it.
 *
 * For scalar types, the analysis is straightforward.
 * For dimensional types, we use integer set analysis:
 *    - A read must intersect with the subset of the definition.
 *    - A new definition must supersede the previous definition.
 */
class DataDependencyAnalysis : public Analysis {
    friend class AnalysisManager;

private:
    structured_control_flow::Sequence& node_;
    std::unordered_map<std::string, std::unordered_map<User*, std::unordered_set<User*>>> results_;

    std::unordered_map<structured_control_flow::StructuredLoop*, std::unordered_map<std::string, LoopCarriedDependency>>
        loop_carried_dependencies_;

    bool loop_depends(
        User& previous,
        User& current,
        analysis::AssumptionsAnalysis& assumptions_analysis,
        structured_control_flow::StructuredLoop& loop
    );

    bool supersedes(User& previous, User& current, analysis::AssumptionsAnalysis& assumptions_analysis);

    bool supersedes_restrictive(User& previous, User& current, analysis::AssumptionsAnalysis& assumptions_analysis);


    bool intersects(User& previous, User& current, analysis::AssumptionsAnalysis& assumptions_analysis);

public:
    DataDependencyAnalysis(StructuredSDFG& sdfg);

    DataDependencyAnalysis(StructuredSDFG& sdfg, structured_control_flow::Sequence& node);

    void run(analysis::AnalysisManager& analysis_manager) override;

    /****** Visitor API ******/

    void visit_block(
        analysis::Users& users,
        analysis::AssumptionsAnalysis& assumptions_analysis,
        structured_control_flow::Block& block,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_for(
        analysis::Users& users,
        analysis::AssumptionsAnalysis& assumptions_analysis,
        structured_control_flow::StructuredLoop& for_loop,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_if_else(
        analysis::Users& users,
        analysis::AssumptionsAnalysis& assumptions_analysis,
        structured_control_flow::IfElse& if_loop,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_while(
        analysis::Users& users,
        analysis::AssumptionsAnalysis& assumptions_analysis,
        structured_control_flow::While& while_loop,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_return(
        analysis::Users& users,
        analysis::AssumptionsAnalysis& assumptions_analysis,
        structured_control_flow::Return& return_statement,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    void visit_sequence(
        analysis::Users& users,
        analysis::AssumptionsAnalysis& assumptions_analysis,
        structured_control_flow::Sequence& sequence,
        std::unordered_set<User*>& undefined,
        std::unordered_map<User*, std::unordered_set<User*>>& open_definitions,
        std::unordered_map<User*, std::unordered_set<User*>>& closed_definitions
    );

    /****** Defines & Use ******/

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

    /****** Loop-carried dependencies ******/

    bool available(structured_control_flow::StructuredLoop& loop) const;

    const std::unordered_map<std::string, LoopCarriedDependency>& dependencies(structured_control_flow::StructuredLoop&
                                                                                   loop) const;
};

} // namespace analysis
} // namespace sdfg
