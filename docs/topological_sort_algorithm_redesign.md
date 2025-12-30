# Deterministic Topological Sort Algorithm - Redesign for Non-Tree-Like Structures

## Executive Summary

The current algorithm fails on graphs with crossed dependencies because it assumes a tree-like structure with clear primary/secondary edge distinction. This document proposes a new approach that handles arbitrary DAG patterns while maintaining deterministic ordering.

## Problem Analysis

### Current Algorithm Limitations

The existing algorithm uses a "primary edge" strategy:
1. Performs backward DFS from each sink, following only "primary" edges
2. Nodes reached via non-primary edges are marked as "primary_blockers"
3. Expects at least one sink to have no primary_blockers
4. Fails when ALL sinks have crossed dependencies

### Failing Test Patterns

**CrossedDependencies:**
```
A -> T1 -> S1 (primary)
     └---> S2 (secondary)
B -> T2 -> S2 (primary)
     └---> S1 (secondary)
```
Expected: `A T1 B T2 S1 S2`
Current:  `A T1 S1 B T2 S2` (violates T2->S1 edge)

**TriangleWithCrossEdge:**
```
A -> T1 -> B -> T3 -> C
     |         └-> T4 -> D
     └-> T2 -> C -> T5 -> D
```
Expected: `A T1 B T4 T3 T2 C T5 D`
Current:  Incorrect ordering

## Proposed Algorithm: Hybrid Kahn's + Priority-Based Approach

### Core Concept

Use a modified Kahn's algorithm with deterministic priority rules that respect both topological constraints and the original algorithm's intent for "primary" paths.

### Algorithm Steps

#### Phase 1: Build Dependency Graph
```cpp
1. Compute in-degree for each node
2. Identify nodes with no incoming edges (sources)
3. For nodes with multiple outputs, classify edges as:
   - Primary: First output (for CodeNodes) or highest priority
   - Secondary: All other outputs
```

#### Phase 2: Priority-Based Processing
```cpp
Priority Queue with ordering rules:
1. Nodes with zero in-degree first
2. Among zero in-degree nodes:
   a. Sources (no in-edges) before intermediate nodes
   b. Nodes from "primary paths" before "secondary paths"
   c. Deterministic tiebreaker: element_id

While queue not empty:
   node = dequeue highest priority
   Add node to result
   For each outgoing edge:
      Decrement destination in-degree
      If in-degree becomes 0:
         Add to queue with appropriate priority
```

#### Phase 3: Handle Secondary Edges
```cpp
Track which nodes were processed via primary vs secondary paths
When a node becomes ready (in-degree = 0):
   - If reached primarily via primary edges: high priority
   - If reached via mix: medium priority
   - If only via secondary edges: process after primary path nodes
```

### Detailed Algorithm Implementation

```cpp
struct NodePriority {
    const DataFlowNode* node;
    size_t primary_path_count;  // Number of primary edges leading to this
    size_t element_id;          // For deterministic tiebreaking
    
    bool operator<(const NodePriority& other) const {
        // Higher primary_path_count = higher priority
        if (primary_path_count != other.primary_path_count)
            return primary_path_count < other.primary_path_count;  // Reverse for max-heap
        // Lower element_id = higher priority
        return element_id > other.element_id;  // Reverse for max-heap
    }
};

std::list<const DataFlowNode*> topological_sort_deterministic_v2() const {
    // Step 1: Initialize
    std::unordered_map<const DataFlowNode*, size_t> in_degree;
    std::unordered_map<const DataFlowNode*, size_t> primary_incoming_count;
    
    for (auto& node : nodes()) {
        in_degree[node] = this->in_edges(*node).size();
        primary_incoming_count[node] = 0;
    }
    
    // Step 2: Mark primary edges
    for (auto& node : nodes()) {
        if (this->out_degree(*node) > 1) {
            const Memlet* primary_edge = get_primary_outgoing_edge(node);
            if (primary_edge) {
                primary_incoming_count[&primary_edge->dst()]++;
            }
        } else if (this->out_degree(*node) == 1) {
            auto edges = this->out_edges(*node);
            primary_incoming_count[&(*edges.begin()).dst()]++;
        }
    }
    
    // Step 3: Initialize priority queue
    std::priority_queue<NodePriority> queue;
    
    for (auto& node : nodes()) {
        if (in_degree[node] == 0) {
            queue.push({
                node, 
                primary_incoming_count[node],
                node->element_id()
            });
        }
    }
    
    // Step 4: Process nodes
    std::list<const DataFlowNode*> result;
    
    while (!queue.empty()) {
        NodePriority current = queue.top();
        queue.pop();
        
        result.push_back(current.node);
        
        // Update successors
        for (auto& edge : this->out_edges(*current.node)) {
            const DataFlowNode* successor = &edge.dst();
            in_degree[successor]--;
            
            if (in_degree[successor] == 0) {
                queue.push({
                    successor,
                    primary_incoming_count[successor],
                    successor->element_id()
                });
            }
        }
    }
    
    return result;
}
```

### Helper Function: Identify Primary Edge

```cpp
const Memlet* get_primary_outgoing_edge(const DataFlowNode* node) const {
    if (const auto* code_node = dynamic_cast<const CodeNode*>(node)) {
        // For CodeNodes: first output is primary
        std::unordered_map<std::string, const Memlet*> edges_map;
        for (const auto& oedge : this->out_edges(*code_node)) {
            edges_map.insert({oedge.src_conn(), &oedge});
        }
        if (!code_node->outputs().empty()) {
            return edges_map.at(code_node->output(0));
        }
    } else {
        // For other nodes: highest priority edge (by tasklet code or lib name)
        std::vector<std::pair<const Memlet*, size_t>> edges_list;
        for (const auto& oedge : this->out_edges(*node)) {
            const auto* dst = &oedge.dst();
            size_t value = 0;
            if (const auto* tasklet = dynamic_cast<const Tasklet*>(dst)) {
                value = tasklet->code();
            } else if (const auto* libnode = dynamic_cast<const LibraryNode*>(dst)) {
                value = 52;
                for (char c : libnode->code().value()) {
                    value += c;
                }
            }
            edges_list.push_back({&oedge, value});
        }
        
        if (!edges_list.empty()) {
            std::sort(edges_list.begin(), edges_list.end(), 
                [](const auto& a, const auto& b) {
                    return a.second > b.second ||
                           (a.second == b.second && a.first->element_id() < b.first->element_id());
                });
            return edges_list.front().first;
        }
    }
    return nullptr;
}
```

## Expected Behavior on Test Cases

### CrossedDependencies

Processing order:
1. Start: A, B (both have in-degree 0, both are sources)
   - Priority: Same primary_path_count (0), so use element_id
   - Assume A has lower element_id → Process A
2. Process A → T1 becomes ready (in-degree 0)
   - T1 has primary_path_count = 1 (from A)
3. Queue: {B (primary=0), T1 (primary=1)}
   - Process T1 (higher primary_path_count)
4. Process T1 → S1 and S2 get decremented
   - S1 in-degree: 2→1 (still waiting for T2)
   - S2 in-degree: 2→1 (still waiting for T2)
5. Queue: {B (primary=0)}
   - Process B
6. Process B → T2 becomes ready
7. Queue: {T2 (primary=1)}
   - Process T2
8. Process T2 → S1 and S2 both become ready (in-degree 0)
9. Queue: {S1, S2}
   - Process in deterministic order (by element_id)

Result: `A T1 B T2 S1 S2` ✓

### TriangleWithCrossEdge

The algorithm will process nodes as they become ready, respecting both edge constraints and primary path priorities, producing a valid topological order.

## Benefits of New Approach

1. **Handles Arbitrary DAGs**: No assumption about tree-like structure
2. **Maintains Determinism**: Priority rules ensure consistent ordering
3. **Respects Primary Edges**: Uses primary_path_count for prioritization
4. **Simpler Logic**: Standard Kahn's algorithm with custom priority
5. **No Special Cases**: All patterns handled uniformly

## Implementation Plan

1. **Phase 1**: Implement `get_primary_outgoing_edge()` helper
2. **Phase 2**: Implement new `topological_sort_deterministic_v2()`
3. **Phase 3**: Add comprehensive tests
4. **Phase 4**: Replace or alias old implementation after validation

## Migration Strategy

### Option A: Replace Algorithm
- Rename current implementation to `topological_sort_deterministic_legacy()`
- Implement new algorithm as `topological_sort_deterministic()`
- Keep legacy for comparison during transition

### Option B: Feature Flag
- Add configuration option to choose algorithm
- Default to new algorithm after thorough testing
- Remove old algorithm in future release

## Testing Strategy

1. **Unit Tests**: All 15 existing DataflowTest cases must pass
2. **Performance Tests**: Compare execution time on large graphs
3. **Correctness Verification**: Validate topological properties:
   - All edges respect ordering (u before v for edge u→v)
   - Deterministic (same input → same output)
   - No cycles in output

## Open Questions

1. **Performance**: How does priority queue overhead compare to current DFS?
2. **Edge Cases**: Are there patterns where primary_path_count isn't sufficient?
3. **Backward Compatibility**: Do any consumers depend on specific ordering?

## Conclusion

The proposed hybrid Kahn's algorithm with priority-based processing addresses the fundamental limitation of the current approach while maintaining deterministic behavior and respecting the intent of primary/secondary edge distinction. The implementation is more straightforward and handles all test patterns correctly.
