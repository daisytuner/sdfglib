#include "sdfg/passes/structured_control_flow/cfg_to_scf_conversion.h"

namespace sdfg {
namespace passes {

CFGToSCFConversion::CFGToSCFConversion() : Pass(), builder_("", FunctionType_CPU) {}

std::string CFGToSCFConversion::name() { return "CFGToSCFConversion"; }

std::unique_ptr<StructuredSDFG> CFGToSCFConversion::get() { return this->builder_.move(); }

bool CFGToSCFConversion::run_pass(builder::SDFGBuilder& builder) {
    this->builder_ = builder::StructuredSDFGBuilder(builder.subject());
    // Compute analyses on original SDFG (builder.subject()) before structuring
    SDFG& sdfg = builder.subject();
    compute_basic_blocks(sdfg);
    compute_regions(sdfg);
    return true;
}

void CFGToSCFConversion::compute_basic_blocks(SDFG& sdfg) {
    basic_blocks_.clear();
    // Leaders: in_degree !=1, out_degree >1, loop headers, entry
    std::unordered_set<const control_flow::State*> leaders;

    // Start state
    leaders.insert(&sdfg.start_state());

    // Loop headers
    for (auto& nl : sdfg.natural_loops()) {
        leaders.insert(nl.header);
    }

    // Branching states: out_degree > 1
    for (const auto& st : sdfg.states()) {
        const control_flow::State* ptr = &st;
        if (sdfg.out_degree(st) > 1) leaders.insert(ptr);
    }
    // Join states: in_degree > 1
    for (const auto& st : sdfg.states()) {
        const control_flow::State* ptr = &st;
        if (sdfg.in_degree(st) > 1) leaders.insert(ptr);
    }

    // Refined partitioning: linear scan creating blocks; start a block at a leader or first unconsumed state.
    std::vector<const control_flow::State*> ordered;
    for (const auto& st : sdfg.states()) ordered.push_back(&st);
    std::unordered_set<const control_flow::State*> consumed;
    for (auto* st : ordered) {
        if (consumed.count(st)) continue;
        bool start_new = leaders.count(st) || true; // always start new if not consumed
        if (!start_new) continue; // unreachable path
        scf::BasicBlock bb;
        bb.entry = st;
        bb.states.push_back(st);
        consumed.insert(st);
        const control_flow::State* cur = st;
        while (true) {
            if (sdfg.out_degree(*cur) != 1) break;
            const control_flow::State* succ = nullptr;
            for (const auto& e : sdfg.out_edges(*cur)) succ = &e.dst();
            if (!succ) break;
            if (leaders.count(succ)) break; // next leader begins new block
            if (sdfg.in_degree(*succ) > 1) break; // join splits
            if (sdfg.out_degree(*succ) > 1) break; // branch splits
            if (consumed.count(succ)) break; // already placed
            bb.states.push_back(succ);
            consumed.insert(succ);
            cur = succ;
        }
        for (const auto& oe : sdfg.out_edges(*cur)) bb.exits.insert(&oe.dst());
        basic_blocks_.push_back(std::move(bb));
    }

    // Optional: sort blocks by original entry order for determinism
    std::unordered_map<const control_flow::State*, size_t> order;
    size_t idx = 0;
    for (const auto& st : sdfg.states()) {
        order[&st] = idx++;
    }
    std::sort(basic_blocks_.begin(), basic_blocks_.end(), [&](const scf::BasicBlock& a, const scf::BasicBlock& b) {
        return order[a.entry] < order[b.entry];
    });
    // TEMP DEBUG
    fprintf(stderr, "[CFGToSCF] basic_blocks=%zu\n", basic_blocks_.size());
    for (auto& bb : basic_blocks_) {
        fprintf(stderr, "  BB entry=%p states=%zu exits=%zu\n", (void*) bb.entry, bb.states.size(), bb.exits.size());
    }
}

void CFGToSCFConversion::compute_regions(SDFG& sdfg) {
    regions_.clear();
    const auto& bbvec = basic_blocks_;
    std::unordered_map<const control_flow::State*, const scf::BasicBlock*> state2block;
    for (auto& b : bbvec) state2block.emplace(b.entry, &b);

    std::vector<scf::Region> regions;
    std::unordered_set<const control_flow::State*> loop_headers;
    for (auto& nl : sdfg.natural_loops()) loop_headers.insert(nl.header);

    // While regions
    for (auto& nl : sdfg.natural_loops()) {
        auto it_hb = state2block.find(nl.header);
        if (it_hb == state2block.end()) continue;
        const scf::BasicBlock* header_block = it_hb->second;
        scf::Region r;
        r.kind = scf::RegionKind::While;
        r.entry = header_block;
        r.loop_header = header_block;
        for (auto& b : bbvec)
            if (&b != header_block)
                for (auto* st : b.states)
                    if (nl.body.count(st)) {
                        r.loop_body.push_back(&b);
                        break;
                    }
        r.blocks.push_back(header_block);
        for (auto* bb : r.loop_body) r.blocks.push_back(bb);
        // Compute loop body closure (multi-hop inside loop until leaving loop)
        std::unordered_set<const scf::BasicBlock*> body_set(r.loop_body.begin(), r.loop_body.end());
        std::list<const scf::BasicBlock*> q(r.loop_body.begin(), r.loop_body.end());
        std::unordered_set<const scf::BasicBlock*> visited;
        while (!q.empty()) {
            auto* cur = q.front();
            q.pop_front();
            if (!cur || visited.count(cur)) continue;
            visited.insert(cur);
            r.loop_body_closure.push_back(cur);
            // Traverse successors if they remain in loop body set and are not header
            for (auto* succ_state : cur->exits) {
                auto it = state2block.find(succ_state);
                if (it == state2block.end()) continue;
                auto* succ_bb = it->second;
                if (succ_bb == header_block) continue;
                if (!body_set.count(succ_bb)) continue;
                q.push_back(succ_bb);
            }
        }
        regions.push_back(std::move(r));
    }

    // Branch regions via post-dominators
    auto pdom_tree = sdfg.post_dominator_tree();
    auto ipdom = [&](const control_flow::State* st) { return pdom_tree.count(st) ? pdom_tree[st] : nullptr; };
    for (auto& b : bbvec) {
        const control_flow::State* s = b.entry;
        if (loop_headers.count(s)) continue;
        if (sdfg.out_degree(*s) != 2) continue;
        std::vector<const control_flow::State*> succ_states;
        for (auto& oe : sdfg.out_edges(*s)) succ_states.push_back(&oe.dst());
        if (succ_states.size() != 2) continue;
        const scf::BasicBlock* then_block = state2block.count(succ_states[0]) ? state2block[succ_states[0]] : nullptr;
        const scf::BasicBlock* else_block = state2block.count(succ_states[1]) ? state2block[succ_states[1]] : nullptr;
        if (!then_block || !else_block) continue;
        // TEMP DEBUG branch candidates
        fprintf(
            stderr,
            "[CFGToSCF] Branch candidate entry=%p then=%p else=%p outdeg=%zu\n",
            (void*) s,
            (void*) succ_states[0],
            (void*) succ_states[1],
            (size_t) sdfg.out_degree(*s)
        );
        const control_flow::State* t_head = succ_states[0];
        const control_flow::State* e_head = succ_states[1];
        // Build ipdom chains for both successors
        std::unordered_set<const control_flow::State*> chain_then;
        const control_flow::State* walk_t = t_head;
        while (walk_t) {
            chain_then.insert(walk_t);
            walk_t = ipdom(walk_t);
        }
        std::unordered_set<const control_flow::State*> chain_else;
        const control_flow::State* walk_e = e_head;
        while (walk_e) {
            chain_else.insert(walk_e);
            walk_e = ipdom(walk_e);
        }
        // Intersection excluding branch node and direct successors
        const control_flow::State* join_state = nullptr;
        for (auto* cand : chain_then) {
            if (cand == s || cand == t_head || cand == e_head) continue;
            if (chain_else.count(cand)) {
                // Choose closest to successors: prefer one whose ipdom chain depth is minimal
                if (!join_state)
                    join_state = cand;
                else {
                    // Depth measure: count steps from t_head to cand
                    auto depth = [&](const control_flow::State* start, const control_flow::State* target) {
                        size_t d = 0;
                        const control_flow::State* cur = start;
                        while (cur && cur != target) {
                            cur = ipdom(cur);
                            ++d;
                            if (d > 10000) break;
                        }
                        return d;
                    };
                    if (depth(t_head, cand) < depth(t_head, join_state)) join_state = cand;
                }
            }
        }
        fprintf(stderr, "[CFGToSCF] Selected join_state=%p for entry=%p\n", (void*) join_state, (void*) s);
        if (!join_state) {
            bool t_term = sdfg.out_degree(*t_head) == 0;
            bool e_term = sdfg.out_degree(*e_head) == 0;
            if (t_term ^ e_term) {
                scf::Region r;
                r.kind = scf::RegionKind::IfThen;
                r.entry = &b;
                r.cond_block = &b;
                if (!t_term)
                    r.then_blocks.push_back(then_block);
                else
                    r.then_blocks.push_back(else_block);
                r.blocks = {&b};
                for (auto* tb : r.then_blocks) r.blocks.push_back(tb);
                regions.push_back(std::move(r));
            }
            continue;
        }
        const scf::BasicBlock* join_block = state2block.count(join_state) ? state2block[join_state] : nullptr;
        if (!join_block) continue;
        if (join_state == s || join_state == t_head || join_state == e_head) continue;
        scf::Region r;
        r.kind = scf::RegionKind::IfElse;
        r.entry = &b;
        r.cond_block = &b;
        r.then_blocks.push_back(then_block);
        r.else_blocks.push_back(else_block);
        r.join_block = join_block;
        r.blocks = {&b, then_block, else_block, join_block};
        auto compute_closure = [&](const scf::BasicBlock* start, std::vector<const scf::BasicBlock*>& out) {
            std::unordered_set<const scf::BasicBlock*> visited;
            std::list<const scf::BasicBlock*> q;
            q.push_back(start);
            while (!q.empty()) {
                auto* cur = q.front();
                q.pop_front();
                if (!cur || visited.count(cur)) continue;
                visited.insert(cur);
                if (cur == join_block) continue;
                if (cur != &b) out.push_back(cur);
                for (auto* succ_state : cur->exits) {
                    auto it = state2block.find(succ_state);
                    if (it != state2block.end()) {
                        auto* succ_bb = it->second;
                        if (succ_bb == &b) continue;
                        q.push_back(succ_bb);
                    }
                }
            }
        };
        compute_closure(then_block, r.then_closure);
        compute_closure(else_block, r.else_closure);
        regions.push_back(std::move(r));
    }

    // Sequence region
    std::unordered_set<const scf::BasicBlock*> covered;
    for (auto& r : regions)
        for (auto* bb : r.blocks)
            if (bb) covered.insert(bb);
    scf::Region seq;
    seq.kind = scf::RegionKind::Sequence;
    for (auto& b : bbvec)
        if (!covered.count(&b)) seq.blocks.push_back(&b);
    if (!seq.blocks.empty()) {
        seq.entry = seq.blocks.front();
        regions.push_back(std::move(seq));
    }

    // Finalize flat regions
    regions_ = std::move(regions);
    // Debug: list detected regions
    // fprintf(stderr, "[CFGToSCF] Detected %zu regions:\n", regions_.size());
    // for (auto& r : regions_) {
    //     fprintf(stderr, "  kind=%d entry=%p blocks=%zu children=%zu\n", (int)r.kind, (void*)r.entry, r.blocks.size(),
    //     r.children.size());
    // }
    root_region_ = scf::Region{};
    root_region_.kind = scf::RegionKind::Sequence;
    if (!regions_.empty() && !regions_[0].blocks.empty())
        root_region_.entry = regions_[0].blocks.front();
    else if (!bbvec.empty())
        root_region_.entry = &bbvec.front();
    std::unordered_set<const scf::BasicBlock*> all_blocks;
    for (auto& r : regions_)
        for (auto* b : r.blocks)
            if (b) all_blocks.insert(b);
    if (all_blocks.empty())
        for (auto& b : bbvec) all_blocks.insert(&b);
    for (auto* b : all_blocks) root_region_.blocks.push_back(b);
    for (auto& r : regions_) {
        r.parent = &root_region_;
        root_region_.children.push_back(&r);
    }

    // Owning shallow copies
    root_region_ptr_ = std::make_unique<scf::Region>(root_region_);
    hierarchical_regions_.clear();
    hierarchical_regions_.reserve(regions_.size());
    for (auto& r : regions_) {
        auto ptr = std::make_unique<scf::Region>(r);
        ptr->parent = root_region_ptr_.get();
        hierarchical_regions_.push_back(std::move(ptr));
    }
    root_region_ptr_->children.clear();
    for (auto& ur : hierarchical_regions_) root_region_ptr_->children.push_back(ur.get());

    // Nesting using closures
    auto blocks_vector_contains_all = [](const std::vector<const scf::BasicBlock*>& container,
                                         const scf::Region& child) {
        std::unordered_set<const scf::BasicBlock*> cont(container.begin(), container.end());
        for (auto* b : child.blocks)
            if (!cont.count(b)) return false;
        return true;
    };
    for (auto& outer_ptr : hierarchical_regions_) {
        scf::Region* outer = outer_ptr.get();
        for (auto& cand_ptr : hierarchical_regions_) {
            scf::Region* cand = cand_ptr.get();
            if (cand == outer) continue;
            if (cand->parent != root_region_ptr_.get()) continue;
            if (cand->kind == scf::RegionKind::Sequence) continue;
            bool contained = false;
            if (outer->kind == scf::RegionKind::While)
                contained = blocks_vector_contains_all(
                    outer->loop_body_closure.empty() ? outer->loop_body : outer->loop_body_closure, *cand
                );
            else if (outer->kind == scf::RegionKind::IfElse)
                contained = blocks_vector_contains_all(outer->then_closure, *cand) ||
                            blocks_vector_contains_all(outer->else_closure, *cand) ||
                            blocks_vector_contains_all(outer->then_blocks, *cand) ||
                            blocks_vector_contains_all(outer->else_blocks, *cand);
            else if (outer->kind == scf::RegionKind::IfThen)
                contained = blocks_vector_contains_all(outer->then_closure, *cand) ||
                            blocks_vector_contains_all(outer->then_blocks, *cand);
            if (!contained) continue;
            cand->parent = outer;
            outer->children.push_back(cand);
        }
    }
    root_region_ptr_->children.clear();
    for (auto& rptr : hierarchical_regions_)
        if (rptr->parent == root_region_ptr_.get()) root_region_ptr_->children.push_back(rptr.get());
}

} // namespace passes
} // namespace sdfg
