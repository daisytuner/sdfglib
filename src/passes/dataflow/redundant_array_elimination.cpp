#include "sdfg/passes/dataflow/redundant_array_elimination.h"

#include "sdfg/analysis/users.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace passes {

RedundantArrayElimination::RedundantArrayElimination()
    : Pass() {

      };

std::string RedundantArrayElimination::name() { return "RedundantArrayElimination"; };

bool RedundantArrayElimination::run_pass(builder::StructuredSDFGBuilder& builder,
                                         analysis::AnalysisManager& analysis_manager) {
    bool applied = false;

    auto& sdfg = builder.subject();
    auto& users = analysis_manager.get<analysis::Users>();
    for (auto& name : sdfg.containers()) {
        // Criterion: Only transients
        if (!sdfg.is_transient(name)) {
            continue;
        }
        if (!users.moves(name).empty() || !users.views(name).empty()) {
            continue;
        }
        if (users.writes(name).size() != 1) {
            continue;
        }
        std::unique_ptr<types::IType> type = sdfg.type(name).clone();
        if (dynamic_cast<const types::Array*>(type.get()) == nullptr) {
            continue;
        }

        // Criterion: Data must depend on externals
        auto write = users.writes(name)[0];
        auto write_node = dynamic_cast<data_flow::AccessNode*>(write->element());
        auto& graph = write_node->get_parent();
        if (graph.in_degree(*write_node) == 0) {
            continue;
        }
        auto& write_edge = *graph.in_edges(*write_node).begin();
        auto& write_subset = write_edge.subset();

        // Access nodes for write
        std::vector<symbolic::Symbol> input_symbols;
        for (auto& e : graph.in_edges(write_edge.src())) {
            auto& src = e.src();
            if (graph.in_degree(src) != 0) {
                continue;
            }
            auto access_node = dynamic_cast<data_flow::AccessNode*>(&src);
            input_symbols.push_back(symbolic::symbol(access_node->data()));
            for (auto& subset : e.subset()) {
                auto atoms = symbolic::atoms(subset);
                for (auto& atom : atoms) {
                    if (SymEngine::is_a<SymEngine::Symbol>(*atom)) {
                        auto symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                        input_symbols.push_back(symbol);
                    }
                }
            }
        }

        bool has_redundant_dimension = false;
        // Criterion: Data must be an Array
        uint depth = 0;
        while (auto atype = dynamic_cast<const types::Array*>(type.get())) {
            auto subset = write_subset[depth];
            auto atoms = symbolic::atoms(subset);
            bool dependency_exists = false;

            for (auto& atom : atoms) {
                if (SymEngine::is_a<SymEngine::Symbol>(*atom)) {
                    auto symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom);
                    for (auto& input_symbol : input_symbols) {
                        if (symbolic::eq(symbol, input_symbol)) {
                            dependency_exists = true;
                            break;
                        }
                    }
                    if (dependency_exists) {
                        break;
                    }
                }
            }
            if (!dependency_exists) {
                has_redundant_dimension = true;
                break;
            }
            type = atype->element_type().clone();
            depth++;
        }

        if (!has_redundant_dimension) {
            continue;
        }

        // Construct new type
        auto atype = static_cast<const types::Array*>(type.get());
        std::unique_ptr<types::IType> inner_type = atype->element_type().clone();
        std::unique_ptr<types::IType> new_type =
            types::recombine_array_type(sdfg.type(name), depth, *inner_type.get());

        // Replace data type
        builder.change_type(name, *new_type.get());

        // Replace all reads
        for (auto& user : users.reads(name)) {
            auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
            if (access_node == nullptr) {
                throw InvalidSDFGException("RedundantArrayElimination: Expected AccessNode");
            }

            auto& graph = access_node->get_parent();
            for (auto& oedge : graph.out_edges(*access_node)) {
                auto& subset = oedge.subset();
                subset.erase(subset.begin() + depth);
            }
        }
        // Replace all writes
        for (auto& user : users.writes(name)) {
            auto access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
            if (access_node == nullptr) {
                throw InvalidSDFGException("RedundantArrayElimination: Expected AccessNode");
            }

            auto& graph = access_node->get_parent();
            for (auto& iedge : graph.in_edges(*access_node)) {
                auto& subset = iedge.subset();
                subset.erase(subset.begin() + depth);
            }
        }

        applied = true;
    }

    analysis_manager.invalidate_all();

    return applied;
};

}  // namespace passes
}  // namespace sdfg
