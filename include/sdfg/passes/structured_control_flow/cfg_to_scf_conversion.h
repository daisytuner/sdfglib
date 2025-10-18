#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

// Helper structures for SCF conversion
namespace scf {

/**
 * @brief A BasicBlock is a maximal linear chain of states.
 */
struct BasicBlock {
    std::vector<const control_flow::State*> states;

    const control_flow::State* entry;

    std::unordered_set<const control_flow::State*> exits;

    BasicBlock() : entry(nullptr) {}

    BasicBlock(const BasicBlock& other) : states(other.states), entry(other.entry), exits(other.exits) {}

    BasicBlock& operator=(const BasicBlock& other) {
        if (this != &other) {
            this->states = other.states;
            this->entry = other.entry;
            this->exits = other.exits;
        }
        return *this;
    }
};

enum class RegionKind { Sequence, While, IfElse, IfThen, Unstructured };

struct Region {
    RegionKind kind{RegionKind::Unstructured};
    // Entry block for the region
    const BasicBlock* entry{nullptr};
    // All blocks owned by this region (flat for now)
    std::vector<const BasicBlock*> blocks;
    // For While
    const BasicBlock* loop_header{nullptr};
    std::vector<const BasicBlock*> loop_body; // excludes header
    std::vector<const BasicBlock*> loop_body_closure; // reachable blocks inside loop (excluding header)
    // For IfElse
    const BasicBlock* cond_block{nullptr};
    std::vector<const BasicBlock*> then_blocks;
    std::vector<const BasicBlock*> else_blocks;
    const BasicBlock* join_block{nullptr};
    // Hierarchy
    std::vector<const Region*> children; // child regions (nested)
    const Region* parent{nullptr}; // parent region (nullptr for root)
    // Arm closures (all blocks reachable within arm until join, including multi-hop chains)
    std::vector<const BasicBlock*> then_closure;
    std::vector<const BasicBlock*> else_closure;
    // For IfThen treat then_closure as closure of taken branch
};

} // namespace scf

class CFGToSCFConversion : public Pass {
private:
    builder::StructuredSDFGBuilder builder_;
    std::vector<scf::BasicBlock> basic_blocks_;
    std::vector<scf::Region> regions_;
    scf::Region root_region_; // legacy synthetic root (flat)
    // Hierarchical (owning) storage
    std::unique_ptr<scf::Region> root_region_ptr_; // owning root with nested children
    std::vector<std::unique_ptr<scf::Region>> hierarchical_regions_; // owning storage for nested regions

    void compute_basic_blocks(SDFG& sdfg);
    void compute_regions(SDFG& sdfg);

public:
    CFGToSCFConversion();

    std::string name() override;

    bool run_pass(builder::SDFGBuilder& builder) override;

    std::unique_ptr<StructuredSDFG> get();

    const std::vector<scf::BasicBlock>& basic_blocks() const { return basic_blocks_; }
    const std::vector<scf::Region>& regions() const { return regions_; }
    const scf::Region& root_region() const { return root_region_; }
    const scf::Region* hierarchical_root() const { return root_region_ptr_.get(); }
    const std::vector<std::unique_ptr<scf::Region>>& hierarchical_regions() const { return hierarchical_regions_; }
};

} // namespace passes
} // namespace sdfg
