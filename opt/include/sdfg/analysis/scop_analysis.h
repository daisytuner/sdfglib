//===- polly/ScopBuilder.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--- polly/DependenceInfo.h - Polyhedral dependency analysis *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is based on the original source files which were modified for sdfglib.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <isl/aff.h>
#include <isl/ast.h>
#include <isl/ast_build.h>
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/schedule.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_set.h>

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/data_dependency_analysis.h>
#include <sdfg/data_flow/code_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/symbolic/symbolic.h>

namespace sdfg {
namespace analysis {

enum class AccessType { READ, WRITE };

class MemoryAccess {
private:
    AccessType access_type_;

    isl_map* relation_;

    std::string data_;

    const data_flow::Memlet* memlet_;

public:
    MemoryAccess(AccessType access_type, isl_map* relation, const std::string& data, const data_flow::Memlet* memlet);

    ~MemoryAccess() {
        if (relation_) {
            isl_map_free(relation_);
        }
    }

    MemoryAccess(const MemoryAccess&) = delete;
    MemoryAccess& operator=(const MemoryAccess&) = delete;

    AccessType access_type() const { return access_type_; }

    isl_map* relation() const { return relation_; }

    const std::string& data() const { return data_; }

    const data_flow::Memlet* memlet() const { return memlet_; }

    void set_relation(isl_map* relation) {
        if (relation_) {
            isl_map_free(relation_);
        }
        relation_ = relation;
    }

    bool is_reduction_like() const { return false; }
};

class ScopStatement {
private:
    std::string name_;

    isl_set* domain_;

    isl_map* schedule_;

    symbolic::SymbolVec iterators_;

    std::vector<std::unique_ptr<MemoryAccess>> memory_accesses_;

    data_flow::CodeNode* code_node_;

    symbolic::Expression expression_;

public:
    ScopStatement(const std::string& name, isl_set* domain, data_flow::CodeNode* code_node);

    ScopStatement(const std::string& name, isl_set* domain, symbolic::Expression expression);

    ~ScopStatement() {
        this->memory_accesses_.clear();
        if (schedule_) {
            isl_map_free(schedule_);
        }
        if (domain_) {
            isl_set_free(domain_);
        }
    }

    ScopStatement(const ScopStatement&) = delete;
    ScopStatement& operator=(const ScopStatement&) = delete;

    const std::string& name() const { return name_; }

    symbolic::Expression expression() const { return expression_; }

    data_flow::CodeNode* code_node() const { return code_node_; }

    isl_set* domain() const { return domain_; }

    void set_domain(isl_set* domain) {
        if (domain_) {
            isl_set_free(domain_);
        }
        domain_ = isl_set_set_tuple_name(domain, name_.c_str());
    }

    void push_front(const symbolic::Symbol& iterator) { iterators_.insert(iterators_.begin(), iterator); }

    const symbolic::SymbolVec& iterators() const { return iterators_; }

    isl_map* schedule() const { return schedule_; }

    void set_schedule(isl_map* schedule) {
        if (schedule_) {
            isl_map_free(schedule_);
        }
        schedule_ = isl_map_set_tuple_name(schedule, isl_dim_in, name_.c_str());
    }

    void insert(std::unique_ptr<MemoryAccess>& access) { memory_accesses_.push_back(std::move(access)); }

    std::unordered_set<MemoryAccess*> reads() const {
        std::unordered_set<MemoryAccess*> reads;
        for (const auto& access : memory_accesses_) {
            if (access->access_type() == AccessType::READ) {
                reads.insert(access.get());
            }
        }
        return reads;
    }

    std::unordered_set<MemoryAccess*> writes() const {
        std::unordered_set<MemoryAccess*> writes;
        for (const auto& access : memory_accesses_) {
            if (access->access_type() == AccessType::WRITE) {
                writes.insert(access.get());
            }
        }
        return writes;
    }

    std::unordered_set<MemoryAccess*> accesses() const {
        std::unordered_set<MemoryAccess*> accesses;
        for (const auto& access : memory_accesses_) {
            accesses.insert(access.get());
        }
        return accesses;
    }

    // Overload operator for print
    friend std::ostream& operator<<(std::ostream& os, const ScopStatement& stmt) {
        os << "ScopStatement: " << stmt.name_ << "\n";
        char* domain_str = isl_set_to_str(stmt.domain_);
        os << "Domain: " << domain_str << "\n";
        free(domain_str);
        os << "Memory Accesses:\n";
        for (const auto& access : stmt.memory_accesses_) {
            char* relation_str = isl_map_to_str(access->relation());
            os << "  - Type: " << (access->access_type() == AccessType::READ ? "READ" : "WRITE")
               << ", Data: " << access->data() << ", Relation: " << relation_str << "\n";
            free(relation_str);
        }
        return os;
    }
};

class Scop {
private:
    structured_control_flow::ControlFlowNode& node_;

    isl_ctx* ctx_;

    isl_space* param_space_;

    isl_schedule* schedule_tree_;

    isl_union_map* schedule_;

    std::vector<std::unique_ptr<ScopStatement>> statements_;

public:
    Scop(structured_control_flow::ControlFlowNode& node, isl_ctx* ctx, isl_space* param_space);

    ~Scop() {
        this->statements_.clear();

        if (schedule_) {
            isl_union_map_free(schedule_);
        }
        if (schedule_tree_) {
            isl_schedule_free(schedule_tree_);
        }
        if (param_space_) {
            isl_space_free(param_space_);
        }
        isl_ctx_free(ctx_);
    }

    Scop(const Scop&) = delete;
    Scop& operator=(const Scop&) = delete;

    structured_control_flow::ControlFlowNode& node() const { return node_; }

    isl_ctx* ctx() const { return ctx_; }

    isl_space* param_space() const { return param_space_; }

    void insert(std::unique_ptr<ScopStatement>& statement) { statements_.push_back(std::move(statement)); }

    std::vector<ScopStatement*> statements() const {
        std::vector<ScopStatement*> stmt_ptrs;
        for (const auto& stmt : statements_) {
            stmt_ptrs.push_back(stmt.get());
        }
        return stmt_ptrs;
    }

    isl_union_set* domains();

    isl_union_map* schedule();

    isl_schedule* schedule_tree();

    void set_schedule_tree(isl_schedule* schedule);

    std::string ast();

    friend std::ostream& operator<<(std::ostream& os, Scop& scop) {
        os << "Scop:\n";
        for (const auto& stmt : scop.statements_) {
            os << *stmt;
        }
        scop.schedule_tree();
        isl_union_map* schedule = scop.schedule();
        char* schedule_str = isl_union_map_to_str(schedule);
        os << "Schedule:\n" << schedule_str << "\n";
        free(schedule_str);
        return os;
    }
};

class ScopBuilder {
private:
    StructuredSDFG& sdfg_;

    structured_control_flow::ControlFlowNode& node_;

    std::unique_ptr<Scop> scop_;

    std::vector<std::string> dimensions_;

    std::unordered_map<std::string, std::string> dimension_constraints_;

    std::vector<std::string> parameters_;

    std::unordered_map<std::string, std::string> parameter_constraints_;

    std::string generate_subset(
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node,
        const symbolic::MultiExpression& subset
    );

public:
    ScopBuilder(StructuredSDFG& sdfg, structured_control_flow::ControlFlowNode& node);

    std::unique_ptr<Scop> build(analysis::AnalysisManager& analysis_manager);

    void visit(analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node);

    void visit_block(analysis::AnalysisManager& analysis_manager, structured_control_flow::Block& block);

    void visit_sequence(analysis::AnalysisManager& analysis_manager, structured_control_flow::Sequence& sequence);

    void visit_structured_loop(analysis::AnalysisManager& analysis_manager, structured_control_flow::StructuredLoop& loop);
};

class Dependences {
private:
    Scop& scop_;

    std::unordered_map<MemoryAccess*, isl_map*> reduction_dependences_;

    /// The different basic kinds of dependences we calculate.
    isl_union_map* RAW;
    isl_union_map* WAR;
    isl_union_map* WAW;

    /// The special reduction dependences.
    isl_union_map* RED;

    /// The (reverse) transitive closure of reduction dependences.
    isl_union_map* TC_RED;

    void add_privatization_dependences();

    void calculate_dependences(Scop& scop);

    void set_reduction_dependences(MemoryAccess* memory_access, isl_map* deps);

public:
    Dependences(Scop& scop) : scop_(scop), RAW(nullptr), WAR(nullptr), WAW(nullptr), RED(nullptr), TC_RED(nullptr) {
        calculate_dependences(scop_);
    }

    Dependences(const Dependences&) = delete;
    Dependences& operator=(const Dependences&) = delete;

    ~Dependences() {
        if (RAW) isl_union_map_free(RAW);
        RAW = nullptr;
        if (WAR) isl_union_map_free(WAR);
        WAR = nullptr;
        if (WAW) isl_union_map_free(WAW);
        WAW = nullptr;
        if (RED) isl_union_map_free(RED);
        RED = nullptr;
        if (TC_RED) isl_union_map_free(TC_RED);
        TC_RED = nullptr;
        for (auto& pair : reduction_dependences_) {
            isl_map_free(pair.second);
        }
        reduction_dependences_.clear();
    }

    /// The type of the dependences.
    ///
    /// Reduction dependences are separated from RAW/WAW/WAR dependences because
    /// we can ignore them during the scheduling. That's because the order
    /// in which the reduction statements are executed does not matter. However,
    /// if they are executed in parallel we need to take additional measures
    /// (e.g, privatization) to ensure a correct result. The (reverse) transitive
    /// closure of the reduction dependences are used to check for parallel
    /// executed reduction statements during code generation. These dependences
    /// connect all instances of a reduction with each other, they are therefore
    /// cyclic and possibly "reversed".
    enum Type {
        // Write after read
        TYPE_WAR = 1 << 0,

        // Read after write
        TYPE_RAW = 1 << 1,

        // Write after write
        TYPE_WAW = 1 << 2,

        // Reduction dependences
        TYPE_RED = 1 << 3,

        // Transitive closure of the reduction dependences (& the reverse)
        TYPE_TC_RED = 1 << 4,
    };

    Scop& scop() const { return scop_; }

    const isl_ctx* ctx() const { return scop_.ctx(); }

    isl_union_map* dependences(int Kinds) const;

    std::unordered_map<std::string, analysis::LoopCarriedDependency>
    dependencies(const sdfg::structured_control_flow::StructuredLoop& loop) const;

    bool has_valid_dependences() const;

    isl_map* reduction_dependences(MemoryAccess* memory_access) const;

    const std::unordered_map<MemoryAccess*, isl_map*>& reduction_dependences() const {
        return this->reduction_dependences_;
    }

    bool is_parallel(isl_union_map* schedule, isl_pw_aff** min_distance_ptr = nullptr) const;

    bool is_valid(Scop& scop, const std::unordered_map<ScopStatement*, isl_map*>& new_schedule) const;

    bool is_valid(Scop& scop, isl_schedule* schedule) const;
};

class ScopToSDFG {
private:
    Scop& scop_;
    const Dependences& dependences_;
    builder::StructuredSDFGBuilder& builder_;
    std::unordered_map<std::string, ScopStatement*> stmt_map_;

    // AST Traversal
    void visit_node(struct isl_ast_node* node, structured_control_flow::Sequence& scope);

    void visit_for(struct isl_ast_node* node, structured_control_flow::Sequence& scope);

    void visit_if(struct isl_ast_node* node, structured_control_flow::Sequence& scope);

    void visit_block(struct isl_ast_node* node, structured_control_flow::Sequence& scope);

    void visit_mark(struct isl_ast_node* node, structured_control_flow::Sequence& scope);

    void visit_user(struct isl_ast_node* node, structured_control_flow::Sequence& scope);

    // Helpers
    symbolic::Expression convert_expr(struct isl_ast_expr* expr);

    symbolic::Condition convert_cond(struct isl_ast_expr* expr);

public:
    ScopToSDFG(Scop& scop, const Dependences& dependences, builder::StructuredSDFGBuilder& builder);

    structured_control_flow::ControlFlowNode& build(analysis::AnalysisManager& analysis_manager);
};

class ScopAnalysis : public Analysis {
private:
    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::unique_ptr<Scop>> scops_;
    std::unordered_map<const structured_control_flow::ControlFlowNode*, std::unique_ptr<Dependences>> dependences_;

public:
    ScopAnalysis(StructuredSDFG& sdfg);

    void run(analysis::AnalysisManager& analysis_manager) override;

    bool has(const structured_control_flow::ControlFlowNode* node) const;

    Scop& scop(const structured_control_flow::ControlFlowNode* node) const;

    const Dependences& dependences(const structured_control_flow::ControlFlowNode* node) const;
};

} // namespace analysis
} // namespace sdfg
