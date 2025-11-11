#pragma once

#include <deque>
#include <unordered_map>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/mem_access_range_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

struct WorkItem {
    const std::string* var_name;
    bool saw_read = false;
    bool saw_write = false;
    bool undefined = false;
    WorkItem* will_complete;
    std::vector<std::tuple<std::vector<symbolic::Expression>, bool, std::vector<symbolic::Expression>, bool>> dims;

    WorkItem(const std::string* var_name) : var_name(var_name), will_complete(nullptr) {}
};

class MemAccessRangesBuilder {
    friend class MemAccessRanges;

private:
    std::deque<WorkItem*> worklist_;
    std::unordered_map<std::string, MemAccessRange> ranges_;

    StructuredSDFG& sdfg_;
    structured_control_flow::ControlFlowNode& node_;

    Users& users_analysis_;
    AssumptionsAnalysis& assumptions_analysis_;

    void process_workItem(WorkItem* item);

    void process_direct_users(WorkItem* item, bool is_write, std::vector<User*> accesses);

    MemAccessRangesBuilder(
        StructuredSDFG& sdfg,
        structured_control_flow::ControlFlowNode& node,
        Users& users_analysis,
        AssumptionsAnalysis& assumptions_analysis
    )
        : sdfg_(sdfg), node_(node), users_analysis_(users_analysis), assumptions_analysis_(assumptions_analysis) {}
};

} // namespace analysis
} // namespace sdfg
