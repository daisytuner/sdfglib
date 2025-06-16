#pragma once

#include <unordered_map>

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

class MemAccessRange {
    friend class MemAccessRangesBuilder;
private:
    const std::string name_;
    bool saw_read_;
    bool saw_write_;
    bool undefined_;
    std::vector<std::pair<symbolic::Expression, symbolic::Expression>> dims_;

public:
    MemAccessRange(
        const std::string& name,
        bool saw_read,
        bool saw_write,
        bool undefined,
        const std::vector<std::pair<symbolic::Expression, symbolic::Expression>>&& dims
    );

    MemAccessRange(const MemAccessRange& other)
        : name_(other.name_),
          saw_read_(other.saw_read_),
          saw_write_(other.saw_write_),
          undefined_(other.undefined_),
          dims_(other.dims_) {}

    MemAccessRange(MemAccessRange&& other) noexcept
        : name_(std::move(other.name_)),
          saw_read_(other.saw_read_),
          saw_write_(other.saw_write_),
          undefined_(other.undefined_),
          dims_(std::move(other.dims_)) {}

    const std::string& get_name() const;

    bool saw_read() const;
    bool saw_write() const;
    bool is_undefined() const;

    const std::vector<std::pair<symbolic::Expression, symbolic::Expression>>& dims() const;
};

class MemAccessRanges : public Analysis {
    friend class AnalysisManager;

   private:
    structured_control_flow::ControlFlowNode& node_;

    // Graph representation
    graph::Graph graph_;

    std::unordered_map<std::string, MemAccessRange> ranges_;

    
   protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

   public:
    MemAccessRanges(StructuredSDFG& sdfg);

    const MemAccessRange* get(const std::string& varName) const;

};

}  // namespace analysis
}  // namespace sdfg