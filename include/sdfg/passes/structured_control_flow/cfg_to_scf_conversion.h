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

    const BasicBlock* entry{nullptr};
    std::vector<const BasicBlock*> blocks;
    std::unordered_set<const BasicBlock*> closure;

    // While
    const BasicBlock* loop_header{nullptr};
    std::vector<const BasicBlock*> loop_body;

    // IfElse
    const BasicBlock* cond_block{nullptr};
    std::vector<const BasicBlock*> then_blocks;
    std::vector<const BasicBlock*> else_blocks;
    const BasicBlock* join_block{nullptr};

    // Hierarchy
    std::vector<const Region*> children;
    const Region* parent{nullptr};
};

} // namespace scf

class CFGToSCFConversion : public Pass {
private:
    builder::StructuredSDFGBuilder builder_;

    std::vector<std::unique_ptr<scf::BasicBlock>> basic_blocks_;
    std::vector<std::unique_ptr<scf::Region>> regions_;
    std::unique_ptr<scf::Region> region_tree_;

    std::unordered_map<const control_flow::State*, const scf::BasicBlock*> state2block_;

    void partition_basic_blocks(SDFG& sdfg);

    void match_regions(SDFG& sdfg);

    bool generate_structured_sdfg(SDFG& sdfg);

    std::unique_ptr<scf::Region> match_while_region(SDFG& sdfg, const analysis::NaturalLoop& loop);

    std::unique_ptr<scf::Region> match_if_else_region(SDFG& sdfg, const scf::BasicBlock& branch_block);

    void build_region_tree();

public:
    CFGToSCFConversion();

    std::string name() override;

    bool run_pass(builder::SDFGBuilder& builder) override;

    std::unique_ptr<StructuredSDFG> get();

    std::vector<const scf::BasicBlock*> basic_blocks() const {
        std::vector<const scf::BasicBlock*> blocks;
        for (const auto& bb_ptr : basic_blocks_) {
            blocks.push_back(bb_ptr.get());
        }

        return blocks;
    }

    std::vector<const scf::Region*> regions() const {
        std::vector<const scf::Region*> regions;
        for (const auto& r_ptr : regions_) {
            regions.push_back(r_ptr.get());
        }

        return regions;
    }

    const scf::Region* region_tree() const { return region_tree_.get(); }
};

} // namespace passes
} // namespace sdfg
