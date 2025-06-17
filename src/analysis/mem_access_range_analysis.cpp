
#include "sdfg/analysis/mem_access_range_analysis.h"

#include <stdbool.h>
#include <symengine/basic.h>
#include <symengine/functions.h>
#include <symengine/infinity.h>
#include <symengine/number.h>
#include <symengine/symengine_rcp.h>
#include <tuple>
#include <utility>
#include <vector>
#include "sdfg/analysis/mem_access_range_analysis_internal.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visualizer/dot_visualizer.h"

#include "sdfg/symbolic/extreme_values.h"

namespace sdfg {
namespace analysis {

MemAccessRanges::MemAccessRanges(StructuredSDFG& sdfg)
    : Analysis(sdfg), node_(sdfg.root()), graph_() {
}

void MemAccessRanges::run(analysis::AnalysisManager& analysis_manager) {

    auto& users = analysis_manager.get<Users>();
    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();

    auto builder = MemAccessRangesBuilder(sdfg_, users, assumptions_analysis);

    auto& worklist = builder.worklist_;

    // Collect argument names
    for (auto& arg : sdfg_.arguments()) {
        if (sdfg_.type(arg).type_id() != types::TypeID::Scalar) {
            worklist.push_back(new WorkItem{&arg});
        }
    }

    // Collect external names
    for (auto& ext : sdfg_.externals()) {
        if (sdfg_.type(ext).type_id() != types::TypeID::Scalar) {
            worklist.push_back(new WorkItem{&ext});
        }
    }

    // Iterate over all variables and their users
    for (auto* workItem : worklist) {
        builder.process_workItem(workItem);
    }

    this->ranges_ = std::move(builder.ranges_);
}

const MemAccessRange* MemAccessRanges::get(const std::string& varName) const {
    auto res = ranges_.find(varName);
    if (res != ranges_.end()) {
        return &res->second;
    } else {
        return nullptr;
    }
}


MemAccessRange::MemAccessRange(
    const std::string& name,
    bool saw_read,
    bool saw_write,
    bool undefined,
    const std::vector<std::pair<symbolic::Expression, symbolic::Expression>>&& dims
): name_(name), saw_read_(saw_read), saw_write_(saw_write), undefined_(undefined), dims_(dims) {}

const std::string& MemAccessRange::get_name() const {
    return name_;
}

bool MemAccessRange::saw_read() const {
    return saw_read_;
}
bool MemAccessRange::saw_write() const {
    return saw_write_;
}
bool MemAccessRange::is_undefined() const {
    return undefined_;
}

const std::vector<std::pair<symbolic::Expression, symbolic::Expression>>& MemAccessRange::dims() const {
    return dims_;
}

void MemAccessRangesBuilder::process_workItem(WorkItem* item) {
    const auto* varName = item->var_name;
    const auto& type = sdfg_.type(*varName);
    
    const auto& reads = users_.reads(*varName);
    process_direct_users(item, varName, false, reads);

    const auto& writes = users_.writes(*varName);
    process_direct_users(item, varName, true, writes);

    const auto& views = users_.views(*varName);
    if (!views.empty()) {
        std::cerr << "Found views for " << *varName << " => not rangeable!" << std::endl;
        item->undefined = true;
    }

    const auto& moves = users_.moves(*varName);
    if (!moves.empty()) {
        std::cerr << "Found moves for " << *varName << " => not rangeable!" << std::endl;
        item->undefined = true;
    }

    if (!item->dims.empty()) {

        std::vector<std::pair<symbolic::Expression, symbolic::Expression>> finalDims;
        finalDims.reserve(item->dims.size());

        for (auto& dim: item->dims) {
            auto& lowerExprs = std::get<0>(dim);
            bool isLowerUndefined = std::get<1>(dim);
            symbolic::Expression lb = (!lowerExprs.empty() && !isLowerUndefined)? SymEngine::min(lowerExprs) : SymEngine::RCP<const SymEngine::Basic>();
            auto& upperExprs = std::get<2>(dim);
            bool isUpperUndefined = std::get<3>(dim);
            symbolic::Expression ub = (!upperExprs.empty() && !isUpperUndefined)? SymEngine::max(upperExprs) : SymEngine::RCP<const SymEngine::Basic>();

            if (lb.is_null() || ub.is_null()) {
                item->undefined = true;
            }
            if (!lb.is_null() && SymEngine::is_a<SymEngine::Infty>(*lb)) {
                lb = SymEngine::null;
                item->undefined = true;
            }
            if (!ub.is_null() && SymEngine::is_a<SymEngine::Infty>(*ub)) {
                ub = SymEngine::null;
                item->undefined = true;
            }

            finalDims.emplace_back(std::move(lb), std::move(ub));
        }

#if !defined(NDEBUG)
        // std::cout << "RangeAna: " << *varName << ": (" << (item->saw_read? "r" : "") << (item->saw_write? "w" : "") << ") " << (!item->undefined? "ranged" : "undef")
        //           << ", " << finalDims.size() << "D: [\n";
        // for (size_t i = 0; i < finalDims.size(); ++i) {
        //     const auto& dim = finalDims[i];
        //     std::cout << "\t(" << (!dim.first.is_null()? dim.first->__str__() : "null") 
        //               << " .. " << (!dim.second.is_null()? dim.second->__str__() : "null") << ")";
        //     std::cout << "\n";
        // }
        // std::cout << "]" << std::endl;
#endif

        this->ranges_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(*varName),
            std::forward_as_tuple(*varName, item->saw_read, item->saw_write, item->undefined, std::move(finalDims))
        );
    }
}

void MemAccessRangesBuilder::process_direct_users(WorkItem* item, const std::string* varName, bool is_write, std::vector<User*> accesses) {

    const auto accessTypeStr = is_write? "w" : "r";
    for (auto& access : accesses) {
        auto subsets = access->subsets();
        const auto& user_scope = analysis::Users::scope(access);
        auto assums = assumptions_analysis_.get(*user_scope, false);
        auto params = this->sdfg_.parameters();

        item->saw_read |= !is_write;
        item->saw_write |= is_write;

        for (const auto& subset : subsets) {
            auto subsetDims = subset.size();
            item->dims.reserve(subsetDims);
            for (size_t i = item->dims.size(); i < subsetDims; ++i) {
                item->dims.emplace_back(std::make_tuple<std::vector<symbolic::Expression>, bool, std::vector<symbolic::Expression>, bool>({}, false, {}, false));
            }
            int dimIdx = 0;
            for (auto& dim : subset) {
                auto lb = symbolic::minimum(dim, params, assums);
                auto ub = symbolic::maximum(dim, params, assums);

                if (lb.is_null()) {
                    std::get<1>(item->dims[dimIdx]) = true;
                } else {
                    std::get<0>(item->dims[dimIdx]).push_back(lb);
                }
                if (ub.is_null()) {
                    std::get<3>(item->dims[dimIdx]) = true;
                } else {
                    std::get<2>(item->dims[dimIdx]).push_back(ub);
                }

                ++dimIdx;
            }
        }
    }
}

}  // namespace analysis
}  // namespace sdfg
