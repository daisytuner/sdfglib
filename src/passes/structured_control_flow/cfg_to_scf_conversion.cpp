#include "sdfg/passes/structured_control_flow/cfg_to_scf_conversion.h"

namespace sdfg {
namespace passes {

CFGToSCFConversion::CFGToSCFConversion() : Pass(), builder_("", FunctionType_CPU) {}

std::string CFGToSCFConversion::name() { return "CFGToSCFConversion"; }

std::unique_ptr<StructuredSDFG> CFGToSCFConversion::get() { return this->builder_.move(); }

bool CFGToSCFConversion::run_pass(builder::SDFGBuilder& builder) {
    basic_blocks_.clear();
    regions_.clear();
    region_tree_ = nullptr;

    this->builder_ = builder::StructuredSDFGBuilder(builder.subject());
    SDFG& sdfg = builder.subject();

    partition_basic_blocks(sdfg);

    match_regions(sdfg);

    return generate_structured_sdfg(sdfg);
}

void CFGToSCFConversion::partition_basic_blocks(SDFG& sdfg) {
    // Leaders: in_degree !=1, out_degree >1, loop headers, entry
    std::unordered_set<const control_flow::State*> leaders;

    // Start state
    leaders.insert(&sdfg.start_state());

    // Branches (out_degree > 1) and merges (in_degree > 1)
    for (const auto& st : sdfg.states()) {
        if (sdfg.out_degree(st) > 1) {
            leaders.insert(&st);
        }
        if (sdfg.in_degree(st) > 1) {
            leaders.insert(&st);
        }
    }

    // Loop headers
    for (auto& nl : sdfg.natural_loops()) {
        leaders.insert(nl.header);
    }

    std::list<const control_flow::State*> queue = {&sdfg.start_state()};
    std::unordered_set<const control_flow::State*> consumed;
    while (!queue.empty()) {
        auto* st = queue.front();
        queue.pop_front();
        if (consumed.count(st)) {
            continue;
        }
        consumed.insert(st);

        // Start new basic block
        auto bb = std::make_unique<scf::BasicBlock>();
        bb->entry = st;
        bb->states.push_back(st);

        const control_flow::State* cur = st;
        while (true) {
            // If out degree != 1, stop here
            if (sdfg.out_degree(*cur) != 1) {
                break;
            }

            auto& oedge = *sdfg.out_edges(*cur).begin();
            const control_flow::State* succ = &oedge.dst();

            // If successor is a leader, stop here
            if (leaders.count(succ)) {
                break;
            }

            if (consumed.count(succ)) {
                break;
            }

            bb->states.push_back(succ);
            consumed.insert(succ);

            cur = succ;
        }
        for (const auto& oe : sdfg.out_edges(*cur)) {
            bb->exits.insert(&oe.dst());
            queue.push_back(&oe.dst());
        }
        basic_blocks_.push_back(std::move(bb));

        for (auto& st : basic_blocks_.back()->states) {
            this->state2block_.emplace(st, basic_blocks_.back().get());
        }
    }
}

std::unique_ptr<scf::Region> CFGToSCFConversion::match_while_region(SDFG& sdfg, const analysis::NaturalLoop& loop) {
    const scf::BasicBlock* header_block = this->state2block_.at(loop.header);

    auto while_region = std::make_unique<scf::Region>();
    while_region->kind = scf::RegionKind::While;
    while_region->entry = header_block;
    while_region->loop_header = header_block;
    while_region->blocks = {header_block};
    while_region->closure = {header_block};

    // Collect all blocks with state membership in loop body
    for (auto& bb : basic_blocks_) {
        if (bb.get() == header_block) {
            continue;
        }

        for (auto* state : bb->states) {
            if (loop.body.count(state)) {
                while_region->loop_body.push_back(bb.get());
                while_region->closure.insert(bb.get());
                break;
            }
        }
    }

    return while_region;
}

std::unique_ptr<scf::Region> CFGToSCFConversion::match_if_else_region(SDFG& sdfg, const scf::BasicBlock& branch_block) {
    if (sdfg.out_degree(*branch_block.entry) > 2) {
        return nullptr;
    }

    // Branch regions via post-dominators (including loop-internal nesting)
    auto pdom_tree = sdfg.post_dominator_tree();
    auto ipdom = [&](const control_flow::State* st) { return pdom_tree.count(st) ? pdom_tree[st] : nullptr; };

    auto entry = branch_block.entry;
    std::vector<const control_flow::State*> succ_states;
    for (auto& oe : sdfg.out_edges(*entry)) {
        succ_states.push_back(&oe.dst());
    }

    const control_flow::State* t_head = succ_states[0];
    const scf::BasicBlock* then_block = this->state2block_.at(t_head);
    if (!then_block) {
        return nullptr;
    }
    const control_flow::State* e_head = succ_states[1];
    const scf::BasicBlock* else_block = this->state2block_.at(e_head);
    if (!else_block) {
        return nullptr;
    }

    // Pattern 1: Early direct-arm IfThen: one successor directly reaches the other in one hop; treat other as taken
    // arm.
    auto single_successor = [&](const control_flow::State* st) {
        if (!st) return (const control_flow::State*) nullptr;
        if (sdfg.out_degree(*st) != 1) return (const control_flow::State*) nullptr;
        for (auto& oe : sdfg.out_edges(*st)) return &oe.dst();
        return (const control_flow::State*) nullptr;
    };

    const control_flow::State* t_succ = single_successor(t_head);
    const control_flow::State* e_succ = single_successor(e_head);
    if ((t_succ && t_succ == e_head) || (e_succ && e_succ == t_head)) {
        bool then_direct_to_else = (t_succ == e_head);
        const scf::BasicBlock* taken_block = then_direct_to_else ? then_block : else_block;

        auto if_then_region = std::make_unique<scf::Region>();
        if_then_region->kind = scf::RegionKind::IfThen;
        if_then_region->entry = &branch_block;
        if_then_region->cond_block = &branch_block;
        if_then_region->blocks = {&branch_block, taken_block, else_block};
        if_then_region->then_blocks.push_back(taken_block);
        if (then_direct_to_else) {
            if_then_region->join_block = else_block;
        } else {
            if_then_region->join_block = then_block;
        }
        if_then_region->closure = {&branch_block, taken_block, else_block};

        return if_then_region;
    }

    // Pattern 2: Full IfElse detection via post-dominator intersection
    // Find common post-dominator of both branch arms
    const control_flow::State* then_pdom = ipdom(t_head);
    const control_flow::State* else_pdom = ipdom(e_head);

    // Find earliest common post-dominator (join point)
    std::unordered_set<const control_flow::State*> then_pdom_chain;
    const control_flow::State* walk = then_pdom;
    while (walk) {
        then_pdom_chain.insert(walk);
        walk = ipdom(walk);
    }

    const control_flow::State* join_state = else_pdom;
    while (join_state && !then_pdom_chain.count(join_state)) {
        join_state = ipdom(join_state);
    }

    if (join_state && join_state != entry) {
        auto join_block = this->state2block_.at(join_state);

        auto pdom_region = std::make_unique<scf::Region>();
        pdom_region->kind = scf::RegionKind::IfElse;
        pdom_region->entry = &branch_block;
        pdom_region->cond_block = &branch_block;
        pdom_region->join_block = join_block;
        pdom_region->then_blocks.push_back(then_block);
        pdom_region->else_blocks.push_back(else_block);
        pdom_region->blocks = {&branch_block, then_block, else_block, join_block};

        // Collect all blocks in then and else branches up to join
        std::function<void(const scf::BasicBlock*, std::vector<const scf::BasicBlock*>&, const scf::BasicBlock*)>
            collect_branch;
        collect_branch = [&](const scf::BasicBlock* bb,
                             std::vector<const scf::BasicBlock*>& closure,
                             const scf::BasicBlock* stop_block) {
            if (bb == stop_block) {
                return;
            }
            closure.push_back(bb);
            for (auto& oe : sdfg.out_edges(*bb->states.back())) {
                auto succ_bb = this->state2block_.at(&oe.dst());
                collect_branch(succ_bb, closure, stop_block);
            }
        };
        collect_branch(then_block, pdom_region->then_blocks, join_block);
        collect_branch(else_block, pdom_region->else_blocks, join_block);

        return pdom_region;
    }

    return nullptr;
}

void CFGToSCFConversion::match_regions(SDFG& sdfg) {
    regions_.clear();
    std::unordered_map<const control_flow::State*, const scf::BasicBlock*> state2block;
    for (auto& b : basic_blocks_) {
        state2block.emplace(b->entry, b.get());
    }

    std::unordered_set<const control_flow::State*> loop_headers;
    for (auto& nl : sdfg.natural_loops()) {
        loop_headers.insert(nl.header);
    }

    std::unordered_set<const scf::Region*> unstructured_regions;

    // While regions
    for (auto& nl : sdfg.natural_loops()) {
        auto while_region = match_while_region(sdfg, nl);
        if (!while_region) {
            // Treat as unstructured region
            auto unstructured_region = std::make_unique<scf::Region>();
            unstructured_region->kind = scf::RegionKind::Unstructured;
            unstructured_region->entry = this->state2block_.at(nl.header);
            unstructured_region->blocks.push_back(unstructured_region->entry);
            this->regions_.push_back(std::move(unstructured_region));
            unstructured_regions.insert(this->regions_.back().get());
            continue;
        }
        this->regions_.push_back(std::move(while_region));
    }
    // Skip further region matching if unstructured regions detected
    if (!unstructured_regions.empty()) {
        DEBUG_PRINTLN("[          ] Aborting region matching due to unstructured regions detected in While matching.");
        return;
    }

    // IfElse regions
    for (auto& bb : basic_blocks_) {
        // A loop header
        if (loop_headers.count(bb->entry)) {
            continue;
        }
        // Straight line code
        if (sdfg.out_degree(*bb->entry) < 2) {
            continue;
        }

        auto ifelse_region = match_if_else_region(sdfg, *bb);
        if (!ifelse_region) {
            // Treat as unstructured region
            auto unstructured_region = std::make_unique<scf::Region>();
            unstructured_region->kind = scf::RegionKind::Unstructured;
            unstructured_region->entry = bb.get();
            unstructured_region->blocks.push_back(bb.get());
            this->regions_.push_back(std::move(unstructured_region));
            unstructured_regions.insert(this->regions_.back().get());
            continue;
        }
        this->regions_.push_back(std::move(ifelse_region));
    }
    // Skip further region matching if unstructured regions detected
    if (!unstructured_regions.empty()) {
        std::cerr << "[          ] Aborting region matching due to unstructured regions detected in IfElse matching."
                  << std::endl;
        return;
    }

    // Convert remaining basic blocks into sequence regions
    std::unordered_set<const scf::BasicBlock*> covered;
    for (auto& r : this->regions_) {
        for (auto* bb : r->blocks) {
            covered.insert(bb);
        }
    }
    for (auto& bb : basic_blocks_) {
        if (covered.count(bb.get())) {
            continue;
        }
        auto sequence_region = std::make_unique<scf::Region>();
        sequence_region->kind = scf::RegionKind::Sequence;
        sequence_region->entry = bb.get();
        sequence_region->blocks.push_back(bb.get());
        this->regions_.push_back(std::move(sequence_region));
        covered.insert(bb.get());
    }
    // Ensure full coverage
    if (covered.size() != basic_blocks_.size()) {
        std::cerr << "[          ] Aborting region matching due to incomplete basic block coverage." << std::endl;
        return;
    }

    // Debug: list detected regions
    std::cerr << "[          ] Detected regions: " << regions_.size() << std::endl;
    for (auto& r : regions_) {
        std::cerr << "  kind=" << (int) r->kind << " entry=" << (void*) r->entry << " blocks=" << r->blocks.size()
                  << std::endl;
    }

    // Build hierarchical region tree
    build_region_tree();
}

void CFGToSCFConversion::build_region_tree() {
    if (regions_.empty()) {
        region_tree_ = nullptr;
        return;
    }

    // TODO
    this->region_tree_ = std::make_unique<scf::Region>();
    this->region_tree_->kind = scf::RegionKind::Sequence;
    this->region_tree_->entry = regions_.front()->entry;
    for (auto& r : regions_) {
        this->region_tree_->children.push_back(r.get());
    }
}

bool CFGToSCFConversion::generate_structured_sdfg(SDFG& sdfg) {
    if (!region_tree_) {
        return false;
    }

    auto visit_bb = [&](structured_control_flow::Sequence& parent, const scf::BasicBlock* bb) {
        if (!bb) {
            return;
        }

        for (size_t i = 0; i < bb->states.size(); ++i) {
            control_flow::Assignments assigns;
            if (sdfg.out_degree(*bb->states[i]) == 1) {
                auto& e = *sdfg.out_edges(*bb->states[i]).begin();
                assigns = e.assignments();
            }

            builder_.add_block(parent, bb->states[i]->dataflow(), assigns, bb->states[i]->debug_info());
        }
    };

    std::function<bool(const scf::Region*, structured_control_flow::Sequence&)> visit_region;
    visit_region = [&](const scf::Region* reg, structured_control_flow::Sequence& parent_seq) -> bool {
        if (!reg) {
            return false;
        }

        switch (reg->kind) {
            case scf::RegionKind::Sequence: {
                // For sequence regions, process child regions in order, not basic blocks directly
                bool success = true;
                for (auto* child : reg->children) {
                    success &= visit_region(child, parent_seq);
                }

                if (reg->children.empty()) {
                    for (auto* bb : reg->blocks) {
                        visit_bb(parent_seq, bb);
                    }
                }
                return success;
            }
            case scf::RegionKind::While: {
                bool body_success = true;

                if (reg->loop_header) {
                    visit_bb(parent_seq, reg->loop_header);
                }

                auto& while_node = builder_.add_while(parent_seq);
                auto& body = while_node.root();
                for (auto* child : reg->children) {
                    body_success &= visit_region(child, body);
                }
                return body_success;
            }
            case scf::RegionKind::IfElse: {
                auto cond_bb = reg->cond_block;
                visit_bb(parent_seq, cond_bb);

                symbolic::Condition then_cond;
                symbolic::Condition else_cond;

                for (auto& oe : sdfg.out_edges(*cond_bb->states.back())) {
                    auto dst_bb = this->state2block_.at(&oe.dst());
                    if (reg->then_blocks.end() != std::find(reg->then_blocks.begin(), reg->then_blocks.end(), dst_bb)) {
                        then_cond = oe.condition();
                    } else if (reg->else_blocks.end() !=
                               std::find(reg->else_blocks.begin(), reg->else_blocks.end(), dst_bb)) {
                        else_cond = oe.condition();
                    } else {
                        return false;
                    }
                }

                // Create if-else
                auto& ifelse_node = builder_.add_if_else(parent_seq);
                auto& then_branch = builder_.add_case(ifelse_node, then_cond);
                auto& else_branch = builder_.add_case(ifelse_node, else_cond);

                // Recurse into branches
                bool children_success = true;
                for (auto* child : reg->children) {
                    auto subset = [&](const std::vector<const scf::BasicBlock*>& side, const scf::Region* ch) {
                        std::unordered_set<const scf::BasicBlock*> side_set(side.begin(), side.end());
                        for (auto* b : ch->blocks)
                            if (!side_set.count(b)) return false;
                        return true;
                    };
                    if (subset(reg->then_blocks, child))
                        children_success &= visit_region(child, then_branch);
                    else
                        children_success &= visit_region(child, else_branch);
                }

                return children_success;
            }
            case scf::RegionKind::Unstructured: {
                return false;
            }
        }
        return false;
    };

    auto& root = builder_.subject().root();
    bool success = visit_region(region_tree_.get(), root);

    return success;
}

} // namespace passes
} // namespace sdfg
