//===- ScopBuilder.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--- DependenceInfo.cpp - Polyhedral dependency analysis *- C++ -*-===//
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
#include "sdfg/analysis/scop_analysis.h"

#include <isl/constraint.h>
#include <isl/flow.h>
#include <isl/id.h>
#include <isl/local_space.h>
#include <isl/options.h>
#include <isl/val.h>

#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/data_flow/library_nodes/math/math.h>
#include <sdfg/symbolic/utils.h>

namespace sdfg {
namespace analysis {


MemoryAccess::
    MemoryAccess(AccessType access_type, isl_map *relation, const std::string &data, const data_flow::Memlet *memlet)
    : access_type_(access_type), relation_(relation), data_(data), memlet_(memlet) {}

ScopStatement::ScopStatement(const std::string &name, isl_set *domain, data_flow::CodeNode *code_node)
    : name_(name), code_node_(code_node), expression_(SymEngine::null) {
    domain_ = isl_set_set_tuple_name(domain, name.c_str());
    isl_space *space = isl_set_get_space(domain_);
    schedule_ = isl_map_identity(isl_space_map_from_set(space));
}

ScopStatement::ScopStatement(const std::string &name, isl_set *domain, symbolic::Expression expression)
    : name_(name), code_node_(nullptr), expression_(expression) {
    domain_ = isl_set_set_tuple_name(domain, name.c_str());
    isl_space *space = isl_set_get_space(domain_);
    schedule_ = isl_map_identity(isl_space_map_from_set(space));
}

Scop::Scop(structured_control_flow::ControlFlowNode &node, isl_ctx *ctx, isl_space *param_space)
    : node_(node), ctx_(ctx), param_space_(param_space), schedule_tree_(nullptr), schedule_(nullptr) {}

isl_union_set *Scop::domains() {
    isl_space *empty_space = isl_space_params_alloc(this->ctx_, 0);
    isl_union_set *domain = isl_union_set_empty(empty_space);

    for (auto &stmt : this->statements_) {
        domain = isl_union_set_add_set(domain, isl_set_copy(stmt->domain()));
    }

    return domain;
}

isl_union_map *Scop::schedule() { return this->schedule_; }

isl_schedule *Scop::schedule_tree() {
    if (this->schedule_tree_) {
        return this->schedule_tree_;
    }

    // Construct default schedule from statements
    isl_space *empty_space = isl_space_params_alloc(this->ctx_, 0);
    isl_union_map *sched_map = isl_union_map_empty(empty_space);
    for (auto &stmt : this->statements_) {
        sched_map = isl_union_map_add_map(sched_map, isl_map_copy(stmt->schedule()));
    }

    isl_schedule *sched = isl_schedule_from_domain(domains());
    if (!isl_union_map_is_empty(sched_map)) {
        // Ensure consistent dimensionality
        isl_map_list *map_list = isl_union_map_get_map_list(sched_map);
        int n_maps = isl_map_list_n_map(map_list);
        int max_dim = 0;

        for (int i = 0; i < n_maps; ++i) {
            isl_map *map = isl_map_list_get_map(map_list, i);
            int dim = isl_map_dim(map, isl_dim_out);
            if (dim > max_dim) {
                max_dim = dim;
            }
            isl_map_free(map);
        }

        isl_union_map *padded_map = isl_union_map_empty(isl_union_map_get_space(sched_map));
        for (int i = 0; i < n_maps; ++i) {
            isl_map *map = isl_map_list_get_map(map_list, i);
            int dim = isl_map_dim(map, isl_dim_out);
            if (dim < max_dim) {
                map = isl_map_add_dims(map, isl_dim_out, max_dim - dim);
                for (int j = dim; j < max_dim; ++j) {
                    map = isl_map_fix_si(map, isl_dim_out, j, 0);
                }
            }
            padded_map = isl_union_map_add_map(padded_map, map);
        }
        isl_map_list_free(map_list);
        isl_union_map_free(sched_map);
        sched_map = padded_map;

        isl_union_pw_multi_aff *upma = isl_union_pw_multi_aff_from_union_map(sched_map);
        isl_multi_union_pw_aff *mupa = isl_multi_union_pw_aff_from_union_pw_multi_aff(upma);
        sched = isl_schedule_insert_partial_schedule(sched, mupa);
    } else {
        isl_union_map_free(sched_map);
    }

    this->schedule_tree_ = sched;
    this->schedule_ = isl_schedule_get_map(this->schedule_tree_);

    return this->schedule_tree_;
}

void Scop::set_schedule_tree(isl_schedule *schedule) {
    if (this->schedule_tree_) {
        isl_schedule_free(this->schedule_tree_);
        isl_union_map_free(this->schedule_);
    }
    this->schedule_tree_ = schedule;
    this->schedule_ = isl_schedule_get_map(schedule);
}

std::string Scop::ast() {
    isl_schedule *schedule = isl_schedule_copy(this->schedule_tree());
    isl_ast_build *build = isl_ast_build_alloc(this->ctx_);
    isl_ast_node *tree = isl_ast_build_node_from_schedule(build, schedule);
    char *str = isl_ast_node_to_C_str(tree);
    std::string result = "";
    if (str) {
        result = std::string(str);
        free(str);
    }
    isl_ast_node_free(tree);
    isl_ast_build_free(build);
    return result;
}

ScopBuilder::ScopBuilder(StructuredSDFG &sdfg, structured_control_flow::ControlFlowNode &node)
    : sdfg_(sdfg), node_(node), scop_(nullptr) {};

std::unique_ptr<Scop> ScopBuilder::build(analysis::AnalysisManager &analysis_manager) {
    isl_ctx *ctx = isl_ctx_alloc();
    isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
    isl_space *param_space = isl_space_set_alloc(ctx, 0, 0);
    this->scop_ = std::make_unique<Scop>(this->node_, ctx, param_space);

    this->visit(analysis_manager, node_);
    if (this->scop_ == nullptr) {
        return nullptr;
    }

    return std::move(scop_);
}

void ScopBuilder::visit(analysis::AnalysisManager &analysis_manager, structured_control_flow::ControlFlowNode &node) {
    if (this->scop_ == nullptr) {
        return;
    }

    if (auto block = dynamic_cast<structured_control_flow::Block *>(&node)) {
        this->visit_block(analysis_manager, *block);
    } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence *>(&node)) {
        this->visit_sequence(analysis_manager, *sequence);
    } else if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop *>(&node)) {
        this->visit_structured_loop(analysis_manager, *loop);
    } else {
        this->scop_ = nullptr;
        return;
    }
}

void ScopBuilder::visit_sequence(analysis::AnalysisManager &analysis_manager, structured_control_flow::Sequence &sequence) {
    for (size_t i = 0; i < sequence.size(); ++i) {
        this->visit(analysis_manager, sequence.at(i).first);
        if (this->scop_ == nullptr) {
            return;
        }

        auto &transition = sequence.at(i).second;
        size_t j = 0;
        for (const auto &assignment : transition.assignments()) {
            isl_set *domain = isl_set_universe(isl_space_set_alloc(scop_->ctx(), 0, 0));
            auto statement = std::make_unique<ScopStatement>(
                "S_" + std::to_string(transition.element_id()) + "_" + std::to_string(j), domain, assignment.second
            );
            j++;

            AccessType access_type = AccessType::WRITE;
            std::string data = assignment.first->get_name();
            std::string relation;
            if (!this->parameters_.empty()) {
                relation += "[";
                relation += helpers::join(this->parameters_, ", ");
                relation += "] -> ";
            }
            relation += "{ [] -> [0] }";
            isl_map *isl_relation = isl_map_read_from_str(scop_->ctx(), relation.c_str());
            if (!isl_relation) {
                this->scop_ = nullptr;
                return;
            }
            isl_relation = isl_map_set_tuple_name(isl_relation, isl_dim_in, statement->name().c_str());
            auto memory_access = std::make_unique<MemoryAccess>(access_type, isl_relation, data, nullptr);
            statement->insert(memory_access);

            for (auto &sym : symbolic::atoms(assignment.second)) {
                AccessType access_type = AccessType::READ;
                std::string data = sym->get_name();
                isl_map *isl_relation = isl_map_read_from_str(scop_->ctx(), relation.c_str());
                if (!isl_relation) {
                    this->scop_ = nullptr;
                    return;
                }
                isl_relation = isl_map_set_tuple_name(isl_relation, isl_dim_in, statement->name().c_str());
                auto memory_access = std::make_unique<MemoryAccess>(access_type, isl_relation, data, nullptr);
                statement->insert(memory_access);
            }

            this->scop_->insert(statement);
        }
    }
}

void ScopBuilder::visit_block(analysis::AnalysisManager &analysis_manager, structured_control_flow::Block &block) {
    auto &assumptions_analysis = analysis_manager.get<analysis::AssumptionsAnalysis>();
    auto &assumptions = assumptions_analysis.get(block, true);

    auto &graph = block.dataflow();
    for (auto node : graph.topological_sort()) {
        if (auto code_node = dynamic_cast<data_flow::CodeNode *>(node)) {
            if (dynamic_cast<data_flow::LibraryNode *>(code_node)) {
                if (!dynamic_cast<math::cmath::CMathNode *>(code_node)) {
                    this->scop_ = nullptr;
                    return;
                }
            }

            isl_set *domain = isl_set_universe(isl_space_set_alloc(scop_->ctx(), 0, 0));
            auto statement =
                std::make_unique<ScopStatement>("S_" + std::to_string(code_node->element_id()), domain, code_node);

            // Add reads
            for (auto &iedge : graph.in_edges(*node)) {
                if (dynamic_cast<data_flow::ConstantNode *>(&iedge.src())) {
                    continue;
                }

                AccessType access_type = AccessType::READ;
                auto delinearized_subset = symbolic::delinearize(iedge.subset(), assumptions);
                std::string relation = generate_subset(analysis_manager, block, delinearized_subset);
                std::string data = static_cast<data_flow::AccessNode &>(iedge.src()).data();
                isl_map *isl_relation = isl_map_read_from_str(scop_->ctx(), relation.c_str());
                if (!isl_relation) {
                    this->scop_ = nullptr;
                    return;
                }
                isl_relation = isl_map_set_tuple_name(isl_relation, isl_dim_in, statement->name().c_str());
                auto memory_access = std::make_unique<MemoryAccess>(access_type, isl_relation, data, &iedge);
                statement->insert(memory_access);
            }

            // Add writes
            for (auto &oedge : graph.out_edges(*node)) {
                AccessType access_type = AccessType::WRITE;

                std::set<std::string> symbols;
                auto delinearized_subset = symbolic::delinearize(oedge.subset(), assumptions);
                std::string relation = generate_subset(analysis_manager, block, delinearized_subset);
                std::string data = static_cast<data_flow::AccessNode &>(oedge.dst()).data();
                isl_map *isl_relation = isl_map_read_from_str(scop_->ctx(), relation.c_str());
                if (!isl_relation) {
                    this->scop_ = nullptr;
                    return;
                }
                isl_relation = isl_map_set_tuple_name(isl_relation, isl_dim_in, statement->name().c_str());
                auto memory_access = std::make_unique<MemoryAccess>(access_type, isl_relation, data, &oedge);
                statement->insert(memory_access);
            }

            this->scop_->insert(statement);
        }
    }
}

void ScopBuilder::
    visit_structured_loop(analysis::AnalysisManager &analysis_manager, structured_control_flow::StructuredLoop &loop) {
    // Add domain for the statements inside the loop
    std::string indvar = loop.indvar()->__str__();
    this->dimensions_.push_back(indvar);

    std::string init = loop.init()->__str__();
    symbolic::CNF condition_cnf;
    try {
        condition_cnf = symbolic::conjunctive_normal_form(loop.condition());
    } catch (symbolic::CNFException &e) {
        this->scop_ = nullptr;
        return;
    }
    std::vector<std::string> constraints;
    constraints.push_back(indvar + " >= " + init);
    for (auto &clause : condition_cnf) {
        if (clause.size() != 1) {
            this->scop_ = nullptr;
            return;
        }
        for (auto &literal : clause) {
            for (auto &sym : symbolic::atoms(literal)) {
                if (sym->get_name() == indvar) {
                    continue;
                }
                if (this->dimension_constraints_.find(sym->get_name()) != this->dimension_constraints_.end()) {
                    continue;
                }
                if (std::find(this->parameters_.begin(), this->parameters_.end(), sym->get_name()) ==
                    this->parameters_.end()) {
                    this->parameters_.push_back(sym->get_name());
                }
            }

            std::string literal_str = symbolic::constraint_to_isl_str(literal);
            if (literal_str.empty()) {
                this->scop_ = nullptr;
                return;
            }
            constraints.push_back(literal_str);
        }
    }
    auto stride = analysis::LoopAnalysis::stride(&loop);
    if (stride.is_null()) {
        this->scop_ = nullptr;
        return;
    }
    int stride_value = stride->as_int();
    if (stride_value != 1) {
        std::string iter = "__daisy_iterator_" + indvar;
        std::string update_constraint = "exists " + iter + " : " + indvar + " = " + init + " + " + iter + " * " +
                                        std::to_string(stride_value);
        constraints.push_back(update_constraint);
    }
    std::string condition = helpers::join(constraints, " and ");
    dimension_constraints_[indvar] = condition;

    // Construct constraints string.
    std::string set_str;
    if (this->dimensions_.size() > 1 || !this->parameters_.empty()) {
        std::vector<std::string> params = this->parameters_;
        for (int i = 0; i < this->dimensions_.size() - 1; ++i) {
            params.push_back(this->dimensions_[i]);
        }
        set_str += "[ " + helpers::join(params, ", ") + " ] -> ";
    }
    set_str += "{ [" + indvar + "] : " + condition + " }";

    isl_ctx *ctx = scop_->ctx();
    isl_set *loop_bound_base = isl_set_read_from_str(ctx, set_str.c_str());
    if (!loop_bound_base) {
        this->scop_ = nullptr;
        return;
    }
    char *loop_bound_str = isl_set_to_str(loop_bound_base);
    free(loop_bound_str);

    size_t stmt_start = this->scop_->statements().size();
    this->visit_sequence(analysis_manager, loop.root());
    if (this->scop_ == nullptr) {
        return;
    }

    auto stmts = this->scop_->statements();
    for (size_t i = stmt_start; i < stmts.size(); ++i) {
        ScopStatement *stmt = stmts[i];
        stmt->push_front(loop.indvar());

        isl_set *domain = isl_set_copy(stmt->domain());

        // Insert indvar dimension at 0
        domain = isl_set_insert_dims(domain, isl_dim_set, 0, 1);
        domain = isl_set_set_dim_name(domain, isl_dim_set, 0, indvar.c_str());

        // Equate parameter with loop dimension if it exists and project it out
        int param_idx = isl_set_find_dim_by_name(domain, isl_dim_param, indvar.c_str());
        if (param_idx >= 0) {
            isl_space *space = isl_set_get_space(domain);
            isl_local_space *ls = isl_local_space_from_space(space);
            isl_constraint *c = isl_constraint_alloc_equality(ls);
            c = isl_constraint_set_coefficient_si(c, isl_dim_set, 0, 1);
            c = isl_constraint_set_coefficient_si(c, isl_dim_param, param_idx, -1);
            domain = isl_set_add_constraint(domain, c);
            domain = isl_set_project_out(domain, isl_dim_param, param_idx, 1);
        }

        // Prepare loop bound set with compatible dimensions
        isl_set *loop_bound = isl_set_copy(loop_bound_base);
        int n_dims = isl_set_dim(domain, isl_dim_set);
        int inner_dims = n_dims - 1;

        if (inner_dims > 0) {
            loop_bound = isl_set_add_dims(loop_bound, isl_dim_set, inner_dims);

            // Match names for inner dimensions
            for (int j = 0; j < inner_dims; ++j) {
                // The inner dims in domain are at indices 1 + j
                const char *name = isl_set_get_dim_name(domain, isl_dim_set, 1 + j);
                if (name) {
                    loop_bound = isl_set_set_dim_name(loop_bound, isl_dim_set, 1 + j, name);
                }
            }
        }

        domain = isl_set_intersect(domain, loop_bound);
        stmt->set_domain(domain);

        // Update Schedule
        isl_map *schedule = isl_map_copy(stmt->schedule());

        // Insert dimension in input (domain)
        schedule = isl_map_insert_dims(schedule, isl_dim_in, 0, 1);
        schedule = isl_map_set_dim_name(schedule, isl_dim_in, 0, indvar.c_str());

        // Equate parameter with loop dimension if it exists and project it out
        param_idx = isl_map_find_dim_by_name(schedule, isl_dim_param, indvar.c_str());
        if (param_idx >= 0) {
            isl_space *space = isl_map_get_space(schedule);
            isl_local_space *ls = isl_local_space_from_space(space);
            isl_constraint *c = isl_constraint_alloc_equality(ls);
            c = isl_constraint_set_coefficient_si(c, isl_dim_in, 0, 1);
            c = isl_constraint_set_coefficient_si(c, isl_dim_param, param_idx, -1);
            schedule = isl_map_add_constraint(schedule, c);
            schedule = isl_map_project_out(schedule, isl_dim_param, param_idx, 1);
        }

        // Insert dimension in output (range/time)
        schedule = isl_map_insert_dims(schedule, isl_dim_out, 0, 1);
        schedule = isl_map_set_dim_name(schedule, isl_dim_out, 0, indvar.c_str());

        // Equate input[0] and output[0]
        schedule = isl_map_equate(schedule, isl_dim_in, 0, isl_dim_out, 0);

        stmt->set_schedule(schedule);

        for (auto access : stmt->accesses()) {
            isl_map *relation = isl_map_copy(access->relation());

            int dom_n = isl_set_dim(stmt->domain(), isl_dim_set);
            int rel_n_in = isl_map_dim(relation, isl_dim_in);

            if (rel_n_in < dom_n) {
                relation = isl_map_insert_dims(relation, isl_dim_in, 0, 1);
                relation = isl_map_set_dim_name(relation, isl_dim_in, 0, indvar.c_str());

                // Equate parameter with loop dimension if it exists and project it out
                param_idx = isl_map_find_dim_by_name(relation, isl_dim_param, indvar.c_str());
                if (param_idx >= 0) {
                    isl_space *space = isl_map_get_space(relation);
                    isl_local_space *ls = isl_local_space_from_space(space);
                    isl_constraint *c = isl_constraint_alloc_equality(ls);
                    c = isl_constraint_set_coefficient_si(c, isl_dim_in, 0, 1);
                    c = isl_constraint_set_coefficient_si(c, isl_dim_param, param_idx, -1);
                    relation = isl_map_add_constraint(relation, c);
                    relation = isl_map_project_out(relation, isl_dim_param, param_idx, 1);
                }
            }

            relation = isl_map_set_tuple_name(relation, isl_dim_in, stmt->name().c_str());
            access->set_relation(relation);
        }
    }
    isl_set_free(loop_bound_base);

    this->dimensions_.pop_back();
}

std::string ScopBuilder::generate_subset(
    analysis::AnalysisManager &analysis_manager,
    structured_control_flow::ControlFlowNode &node,
    const symbolic::MultiExpression &subset
) {
    std::string relation = "";
    if (!this->parameters_.empty()) {
        relation += "[ " + helpers::join(this->parameters_, ", ") + " ] -> ";
    }
    relation += "{ [";
    relation += helpers::join(this->dimensions_, ", ");
    relation += "] -> [";
    if (subset.empty()) {
        relation += "0";
    } else {
        for (size_t i = 0; i < subset.size(); ++i) {
            if (i > 0) {
                relation += ", ";
            }
            relation += subset.at(i)->__str__();
        }
    }
    relation += "] }";

    return relation;
}

static isl_map *tag(isl_map *Relation, isl_id *TagId) {
    isl_space *Space = isl_map_get_space(Relation);
    Space = isl_space_drop_dims(Space, isl_dim_out, 0, isl_map_dim(Relation, isl_dim_out));
    Space = isl_space_set_tuple_id(Space, isl_dim_out, TagId);
    isl_multi_aff *Tag = isl_multi_aff_domain_map(Space);
    Relation = isl_map_preimage_domain_multi_aff(Relation, Tag);
    return Relation;
}

static void collectInfo(
    Scop &S,
    isl_union_map *&Read,
    isl_union_map *&MustWrite,
    isl_union_map *&ReductionTagMap,
    isl_union_set *&TaggedStmtDomain
) {
    isl_space *Space = isl_space_copy(S.param_space());
    Read = isl_union_map_empty(isl_space_copy(Space));
    MustWrite = isl_union_map_empty(isl_space_copy(Space));
    ReductionTagMap = isl_union_map_empty(isl_space_copy(Space));
    isl_union_map *StmtSchedule = isl_union_map_empty(isl_space_copy(Space));

    std::unordered_set<std::string> ReductionArrays;
    for (auto stmt : S.statements())
        for (MemoryAccess *MA : stmt->accesses())
            if (MA->is_reduction_like()) ReductionArrays.insert(MA->data());

    for (auto stmt : S.statements()) {
        for (MemoryAccess *MA : stmt->accesses()) {
            isl_set *domcp = isl_set_copy(stmt->domain());
            isl_map *accdom = isl_map_copy(MA->relation());
            accdom = isl_map_set_tuple_name(accdom, isl_dim_out, MA->data().c_str());

            accdom = isl_map_intersect_domain(accdom, domcp);


            if (ReductionArrays.count(MA->data())) {
                // Wrap the access domain and adjust the schedule accordingly.
                //
                // An access domain like
                //   Stmt[i0, i1] -> MemAcc_A[i0 + i1]
                // will be transformed into
                //   [Stmt[i0, i1] -> MemAcc_A[i0 + i1]] -> MemAcc_A[i0 + i1]
                //
                // We collect all the access domains in the ReductionTagMap.
                // This is used in Dependences::calculateDependences to create
                // a tagged Schedule tree.

                ReductionTagMap = isl_union_map_add_map(ReductionTagMap, isl_map_copy(accdom));
                accdom = isl_map_range_map(accdom);
            } else {
                isl_map *StmtScheduleMap = isl_map_copy(stmt->schedule());
                assert(
                    StmtScheduleMap &&
                    "Schedules that contain extension nodes require special "
                    "handling."
                );

                isl_id *id = isl_id_alloc(S.ctx(), MA->data().c_str(), MA);
                accdom = tag(accdom, isl_id_copy(id));
                isl_map *Schedule = tag(StmtScheduleMap, id);
                StmtSchedule = isl_union_map_add_map(StmtSchedule, Schedule);
            }

            if (MA->access_type() == AccessType::READ) {
                Read = isl_union_map_add_map(Read, accdom);
            } else {
                MustWrite = isl_union_map_add_map(MustWrite, accdom);
            }
        }
    }

    StmtSchedule = isl_union_map_intersect_params(StmtSchedule, isl_set_universe(isl_space_copy(Space)));
    TaggedStmtDomain = isl_union_map_domain(StmtSchedule);

    ReductionTagMap = isl_union_map_coalesce(ReductionTagMap);
    Read = isl_union_map_coalesce(Read);
    MustWrite = isl_union_map_coalesce(MustWrite);

    isl_space_free(Space);
}

static isl_union_flow *
buildFlow(isl_union_map *Snk, isl_union_map *Src, isl_union_map *MaySrc, isl_union_map *Kill, isl_schedule *Schedule) {
    isl_union_access_info *AI;

    AI = isl_union_access_info_from_sink(isl_union_map_copy(Snk));
    if (MaySrc) AI = isl_union_access_info_set_may_source(AI, isl_union_map_copy(MaySrc));
    if (Src) AI = isl_union_access_info_set_must_source(AI, isl_union_map_copy(Src));
    if (Kill) AI = isl_union_access_info_set_kill(AI, isl_union_map_copy(Kill));
    AI = isl_union_access_info_set_schedule(AI, isl_schedule_copy(Schedule));
    auto Flow = isl_union_access_info_compute_flow(AI);
    return Flow;
}

static isl_stat get_map_range_space(isl_map *Map, void *User) {
    isl_space **Space = (isl_space **) User;
    if (*Space) {
        isl_map_free(Map);
        return isl_stat_ok;
    }
    *Space = isl_space_range(isl_map_get_space(Map));
    isl_map_free(Map);
    return isl_stat_ok;
}

static isl_stat fix_set_to_zero(isl_set *Set, void *User) {
    isl_union_set **UserUS = (isl_union_set **) User;

    int dims = isl_set_dim(Set, isl_dim_set);
    for (int i = 0; i < dims; ++i) {
        Set = isl_set_fix_si(Set, isl_dim_set, i, 0);
    }

    *UserUS = isl_union_set_union(*UserUS, isl_union_set_from_set(Set));
    return isl_stat_ok;
}

isl_union_map *Dependences::dependences(int Kinds) const {
    assert(has_valid_dependences() && "No valid dependences available");
    isl_space *Space = isl_union_map_get_space(RAW);
    isl_union_map *Deps = isl_union_map_empty(Space);

    if (Kinds & TYPE_RAW) Deps = isl_union_map_union(Deps, isl_union_map_copy(RAW));

    if (Kinds & TYPE_WAR) Deps = isl_union_map_union(Deps, isl_union_map_copy(WAR));

    if (Kinds & TYPE_WAW) Deps = isl_union_map_union(Deps, isl_union_map_copy(WAW));

    if (Kinds & TYPE_RED) Deps = isl_union_map_union(Deps, isl_union_map_copy(RED));

    if (Kinds & TYPE_TC_RED) Deps = isl_union_map_union(Deps, isl_union_map_copy(TC_RED));

    Deps = isl_union_map_coalesce(Deps);
    Deps = isl_union_map_detect_equalities(Deps);
    return Deps;
}

struct DepCallbackInfo {
    std::unordered_map<std::string, analysis::LoopCarriedDependency> *deps;
    analysis::LoopCarriedDependency type;
    const sdfg::structured_control_flow::StructuredLoop *loop;
};

static isl_stat collect_deps(isl_map *bmap, void *user) {
    auto *info = static_cast<DepCallbackInfo *>(user);

    if (info->loop) {
        bool exists_for_dim = false;
        std::string indvar_name = info->loop->indvar()->get_name();

        isl_map *map_for_deltas = isl_map_copy(bmap);
        isl_space *map_space = isl_map_get_space(map_for_deltas);
        isl_space *dom_space = isl_space_domain(isl_space_copy(map_space));
        if (isl_space_is_wrapping(dom_space)) {
            map_for_deltas = isl_map_domain_factor_domain(map_for_deltas);
        }
        isl_space_free(dom_space);

        isl_space *ran_space = isl_space_range(map_space);
        if (isl_space_is_wrapping(ran_space)) {
            map_for_deltas = isl_map_range_factor_domain(map_for_deltas);
        }
        isl_space_free(ran_space);

        isl_space *map_space_final = isl_map_get_space(map_for_deltas);
        isl_space *domain_space = isl_space_domain(isl_space_copy(map_space_final));
        isl_space *range_space = isl_space_range(isl_space_copy(map_space_final));
        isl_space_free(map_space_final);

        if (!isl_space_is_equal(domain_space, range_space)) {
            isl_space_free(domain_space);
            isl_space_free(range_space);
            isl_map_free(map_for_deltas);
            isl_map_free(bmap);
            return isl_stat_ok;
        }
        isl_space_free(domain_space);
        isl_space_free(range_space);

        isl_set *deltas = isl_map_deltas(map_for_deltas);
        int dims = isl_set_dim(deltas, isl_dim_set);
        int dim_idx = -1;

        for (int i = 0; i < dims; ++i) {
            const char *name = isl_set_get_dim_name(deltas, isl_dim_set, i);
            if (name && indvar_name == name) {
                dim_idx = i;
                break;
            }
        }

        if (dim_idx != -1) {
            // Check if there are any dependencies carried by this dimension.
            // This means we check if there exists a dependence vector d such that:
            // d[0...dim_idx-1] = 0 AND d[dim_idx] != 0

            for (int i = 0; i < dim_idx; ++i) {
                deltas = isl_set_fix_si(deltas, isl_dim_set, i, 0);
            }

            isl_set *zero_at_dim = isl_set_universe(isl_set_get_space(deltas));
            zero_at_dim = isl_set_fix_si(zero_at_dim, isl_dim_set, dim_idx, 0);

            if (!isl_set_is_subset(deltas, zero_at_dim)) {
                exists_for_dim = true;
            }
            isl_set_free(zero_at_dim);
        }
        isl_set_free(deltas);

        if (!exists_for_dim) {
            isl_map_free(bmap);
            return isl_stat_ok;
        }
    }

    isl_space *space = isl_map_get_space(bmap);
    isl_space *domain_space = isl_space_domain(space);

    if (isl_space_is_wrapping(domain_space)) {
        isl_space *domain_space_copy = isl_space_copy(domain_space);
        isl_space *unwrapped = isl_space_unwrap(domain_space_copy);
        isl_space *range = isl_space_range(unwrapped);

        if (isl_space_has_tuple_name(range, isl_dim_set)) {
            const char *name = isl_space_get_tuple_name(range, isl_dim_set);
            if (name) {
                std::string sname(name);
                auto &map = *info->deps;
                if (map.find(sname) == map.end()) {
                    map[sname] = info->type;
                } else {
                    if (info->type == analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE) {
                        map[sname] = info->type;
                    }
                }
            }
        }
        isl_space_free(range);
    }

    isl_space_free(domain_space);
    isl_map_free(bmap);
    return isl_stat_ok;
}

std::unordered_map<std::string, analysis::LoopCarriedDependency> Dependences::
    dependencies(const sdfg::structured_control_flow::StructuredLoop &loop) const {
    std::unordered_map<std::string, analysis::LoopCarriedDependency> deps;
    DepCallbackInfo info;
    info.deps = &deps;
    info.loop = &loop;

    if (WAW) {
        info.type = analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE;
        isl_union_map_foreach_map(WAW, collect_deps, &info);
    }

    if (RAW) {
        info.type = analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE;
        isl_union_map_foreach_map(RAW, collect_deps, &info);
    }

    if (WAR) {
        info.type = analysis::LoopCarriedDependency::LOOP_CARRIED_DEPENDENCY_READ_WRITE;
        isl_union_map_foreach_map(WAR, collect_deps, &info);
    }

    return deps;
}

bool Dependences::has_valid_dependences() const { return (RAW != nullptr) && (WAR != nullptr) && (WAW != nullptr); }

isl_map *Dependences::reduction_dependences(MemoryAccess *MA) const { return reduction_dependences_.at(MA); }

void Dependences::set_reduction_dependences(MemoryAccess *memory_access, isl_map *deps) {
    reduction_dependences_[memory_access] = deps;
}

void Dependences::calculate_dependences(Scop &S) {
    isl_union_map *Read, *MustWrite, *ReductionTagMap;
    isl_union_set *TaggedStmtDomain;
    collectInfo(S, Read, MustWrite, ReductionTagMap, TaggedStmtDomain);

    isl_schedule *Schedule = isl_schedule_copy(S.schedule_tree());

    bool has_reductions = !isl_union_map_is_empty(ReductionTagMap);
    if (!has_reductions) {
        isl_union_map_free(ReductionTagMap);
        // Tag the schedule tree if we want fine-grain dependence info
        auto TaggedMap = isl_union_set_unwrap(isl_union_set_copy(TaggedStmtDomain));
        auto Tags = isl_union_map_domain_map_union_pw_multi_aff(TaggedMap);
        Schedule = isl_schedule_pullback_union_pw_multi_aff(Schedule, Tags);
    } else {
        // Extract Reduction tags from the combined access domains in the given
        // SCoP. The result is a map that maps each tagged element in the domain to
        // the memory location it accesses. ReductionTags = {[Stmt[i] ->
        // Array[f(i)]] -> Stmt[i] }
        isl_union_pw_multi_aff *ReductionTags = isl_union_map_domain_map_union_pw_multi_aff(ReductionTagMap);

        // Compute an identity map from each statement in domain to itself.
        // IdentityTags = { [Stmt[i] -> Stmt[i] }
        isl_union_map *IdentityMap = isl_union_set_identity(isl_union_set_copy(TaggedStmtDomain));
        isl_union_pw_multi_aff *IdentityTags = isl_union_pw_multi_aff_from_union_map(IdentityMap);

        isl_union_pw_multi_aff *Tags = isl_union_pw_multi_aff_union_add(ReductionTags, IdentityTags);

        // By pulling back Tags from Schedule, we have a schedule tree that can
        // be used to compute normal dependences, as well as 'tagged' reduction
        // dependences.
        Schedule = isl_schedule_pullback_union_pw_multi_aff(Schedule, Tags);
    }

    isl_union_map *StrictWAW = nullptr;
    {
        RAW = WAW = WAR = RED = nullptr;
        isl_union_map *Write = isl_union_map_copy(MustWrite);

        // We are interested in detecting reductions that do not have intermediate
        // computations that are captured by other statements.
        //
        // Example:
        // void f(int *A, int *B) {
        //     for(int i = 0; i <= 100; i++) {
        //
        //            *-WAR (S0[i] -> S0[i + 1] 0 <= i <= 100)------------*
        //            |                                                   |
        //            *-WAW (S0[i] -> S0[i + 1] 0 <= i <= 100)------------*
        //            |                                                   |
        //            v                                                   |
        //     S0:    *A += i; >------------------*-----------------------*
        //                                        |
        //         if (i >= 98) {          WAR (S0[i] -> S1[i]) 98 <= i <= 100
        //                                        |
        //     S1:        *B = *A; <--------------*
        //         }
        //     }
        // }
        //
        // S0[0 <= i <= 100] has a reduction. However, the values in
        // S0[98 <= i <= 100] is captured in S1[98 <= i <= 100].
        // Since we allow free reordering on our reduction dependences, we need to
        // remove all instances of a reduction statement that have data dependences
        // originating from them.
        // In the case of the example, we need to remove S0[98 <= i <= 100] from
        // our reduction dependences.
        //
        // When we build up the WAW dependences that are used to detect reductions,
        // we consider only **Writes that have no intermediate Reads**.
        //
        // `isl_union_flow_get_must_dependence` gives us dependences of the form:
        // (sink <- must_source).
        //
        // It *will not give* dependences of the form:
        // 1. (sink <- ... <- may_source <- ... <- must_source)
        // 2. (sink <- ... <- must_source <- ... <- must_source)
        //
        // For a detailed reference on ISL's flow analysis, see:
        // "Presburger Formulas and Polyhedral Compilation" - Approximate Dataflow
        //  Analysis.
        //
        // Since we set "Write" as a must-source, "Read" as a may-source, and ask
        // for must dependences, we get all Writes to Writes that **do not flow
        // through a Read**.
        //
        // ScopInfo::checkForReductions makes sure that if something captures
        // the reduction variable in the same basic block, then it is rejected
        // before it is even handed here. This makes sure that there is exactly
        // one read and one write to a reduction variable in a Statement.
        // Example:
        //     void f(int *sum, int A[N], int B[N]) {
        //       for (int i = 0; i < N; i++) {
        //         *sum += A[i]; < the store and the load is not tagged as a
        //         B[i] = *sum;  < reduction-like access due to the overlap.
        //       }
        //     }

        isl_union_flow *Flow = buildFlow(Write, Write, Read, nullptr, Schedule);
        StrictWAW = isl_union_flow_get_must_dependence(Flow);
        isl_union_flow_free(Flow);

        Flow = buildFlow(Read, nullptr, Write, nullptr, Schedule);
        RAW = isl_union_flow_get_may_dependence(Flow);
        isl_union_flow_free(Flow);

        Flow = buildFlow(Write, nullptr, Read, nullptr, Schedule);
        WAR = isl_union_flow_get_may_dependence(Flow);
        isl_union_flow_free(Flow);

        Flow = buildFlow(Write, nullptr, Write, nullptr, Schedule);
        WAW = isl_union_flow_get_may_dependence(Flow);
        isl_union_flow_free(Flow);

        isl_union_map_free(Write);
        isl_union_map_free(MustWrite);
        isl_union_map_free(Read);
        isl_schedule_free(Schedule);

        RAW = isl_union_map_coalesce(RAW);
        WAW = isl_union_map_coalesce(WAW);
        WAR = isl_union_map_coalesce(WAR);

        // End of max_operations scope.
    }

    isl_union_map *STMT_RAW =
        isl_union_map_intersect_domain(isl_union_map_copy(RAW), isl_union_set_copy(TaggedStmtDomain));
    isl_union_map *STMT_WAW =
        isl_union_map_intersect_domain(isl_union_map_copy(WAW), isl_union_set_copy(TaggedStmtDomain));
    isl_union_map *STMT_WAR =
        isl_union_map_intersect_domain(isl_union_map_copy(WAR), isl_union_set_copy(TaggedStmtDomain));
    isl_union_set_free(TaggedStmtDomain);

    // To handle reduction dependences we proceed as follows:
    // 1) Aggregate all possible reduction dependences, namely all self
    //    dependences on reduction like statements.
    // 2) Intersect them with the actual RAW & WAW dependences to the get the
    //    actual reduction dependences. This will ensure the load/store memory
    //    addresses were __identical__ in the two iterations of the statement.
    // 3) Relax the original RAW, WAW and WAR dependences by subtracting the
    //    actual reduction dependences. Binary reductions (sum += A[i]) cause
    //    the same, RAW, WAW and WAR dependences.
    // 4) Add the privatization dependences which are widened versions of
    //    already present dependences. They model the effect of manual
    //    privatization at the outermost possible place (namely after the last
    //    write and before the first access to a reduction location).

    // Step 1)
    RED = isl_union_map_empty(isl_union_map_get_space(RAW));
    for (auto stmt : S.statements()) {
        for (MemoryAccess *MA : stmt->accesses()) {
            if (!MA->is_reduction_like()) {
                continue;
            }
            isl_set *AccDomW = isl_map_wrap(isl_map_copy(MA->relation()));
            isl_map *Identity = isl_map_from_domain_and_range(isl_set_copy(AccDomW), AccDomW);
            RED = isl_union_map_add_map(RED, Identity);
        }
    }

    // Step 2)
    RED = isl_union_map_intersect(RED, isl_union_map_copy(RAW));
    if (isl_union_map_is_empty(RED)) {
        isl_union_map_free(StrictWAW);
    } else {
        RED = isl_union_map_intersect(RED, StrictWAW);
    }

    if (!isl_union_map_is_empty(RED)) {
        // Step 3)
        RAW = isl_union_map_subtract(RAW, isl_union_map_copy(RED));
        WAW = isl_union_map_subtract(WAW, isl_union_map_copy(RED));
        WAR = isl_union_map_subtract(WAR, isl_union_map_copy(RED));

        // Step 4)
        add_privatization_dependences();
    } else {
        TC_RED = isl_union_map_empty(isl_union_map_get_space(RED));
    }

    // RED_SIN is used to collect all reduction dependences again after we
    // split them according to the causing memory accesses. The current assumption
    // is that our method of splitting will not have any leftovers. In the end
    // we validate this assumption until we have more confidence in this method.
    isl_union_map *RED_SIN = isl_union_map_empty(isl_union_map_get_space(RAW));

    // For each reduction like memory access, check if there are reduction
    // dependences with the access relation of the memory access as a domain
    // (wrapped space!). If so these dependences are caused by this memory access.
    // We then move this portion of reduction dependences back to the statement ->
    // statement space and add a mapping from the memory access to these
    // dependences.
    for (auto stmt : S.statements()) {
        for (MemoryAccess *MA : stmt->accesses()) {
            if (!MA->is_reduction_like()) {
                continue;
            }

            isl_set *AccDomW = isl_map_wrap(isl_map_copy(MA->relation()));
            isl_union_map *AccRedDepU =
                isl_union_map_intersect_domain(isl_union_map_copy(TC_RED), isl_union_set_from_set(AccDomW));
            if (isl_union_map_is_empty(AccRedDepU)) {
                isl_union_map_free(AccRedDepU);
                continue;
            }

            isl_map *AccRedDep = isl_map_from_union_map(AccRedDepU);
            RED_SIN = isl_union_map_add_map(RED_SIN, isl_map_copy(AccRedDep));
            AccRedDep = isl_map_zip(AccRedDep);
            AccRedDep = isl_set_unwrap(isl_map_domain(AccRedDep));
            set_reduction_dependences(MA, AccRedDep);
        }
    }

    assert(
        isl_union_map_is_equal(RED_SIN, TC_RED) &&
        "Intersecting the reduction dependence domain with the wrapped access "
        "relation is not enough, we need to loosen the access relation also"
    );
    isl_union_map_free(RED_SIN);

    RAW = isl_union_map_zip(RAW);
    WAW = isl_union_map_zip(WAW);
    WAR = isl_union_map_zip(WAR);
    RED = isl_union_map_zip(RED);
    TC_RED = isl_union_map_zip(TC_RED);

    RAW = isl_union_set_unwrap(isl_union_map_domain(RAW));
    WAW = isl_union_set_unwrap(isl_union_map_domain(WAW));
    WAR = isl_union_set_unwrap(isl_union_map_domain(WAR));
    RED = isl_union_set_unwrap(isl_union_map_domain(RED));
    TC_RED = isl_union_set_unwrap(isl_union_map_domain(TC_RED));

    RAW = isl_union_map_union(RAW, STMT_RAW);
    WAW = isl_union_map_union(WAW, STMT_WAW);
    WAR = isl_union_map_union(WAR, STMT_WAR);

    RAW = isl_union_map_coalesce(RAW);
    WAW = isl_union_map_coalesce(WAW);
    WAR = isl_union_map_coalesce(WAR);
    RED = isl_union_map_coalesce(RED);
    TC_RED = isl_union_map_coalesce(TC_RED);
}

void Dependences::add_privatization_dependences() {
    isl_union_map *PrivRAW, *PrivWAW, *PrivWAR;

    // The transitive closure might be over approximated, thus could lead to
    // dependency cycles in the privatization dependences. To make sure this
    // will not happen we remove all negative dependences after we computed
    // the transitive closure.
    TC_RED = isl_union_map_transitive_closure(isl_union_map_copy(RED), nullptr);

    // FIXME: Apply the current schedule instead of assuming the identity schedule
    //        here. The current approach is only valid as long as we compute the
    //        dependences only with the initial (identity schedule). Any other
    //        schedule could change "the direction of the backward dependences" we
    //        want to eliminate here.
    isl_union_set *UDeltas = isl_union_map_deltas(isl_union_map_copy(TC_RED));
    isl_union_set *Universe = isl_union_set_universe(isl_union_set_copy(UDeltas));
    isl_union_set *Zero = isl_union_set_empty(isl_union_set_get_space(Universe));

    isl_union_set_foreach_set(Universe, fix_set_to_zero, &Zero);
    isl_union_set_free(Universe);

    isl_union_map *NonPositive = isl_union_set_lex_le_union_set(UDeltas, Zero);

    TC_RED = isl_union_map_subtract(TC_RED, NonPositive);

    TC_RED = isl_union_map_union(TC_RED, isl_union_map_reverse(isl_union_map_copy(TC_RED)));
    TC_RED = isl_union_map_coalesce(TC_RED);

    isl_union_map **Maps[] = {&RAW, &WAW, &WAR};
    isl_union_map **PrivMaps[] = {&PrivRAW, &PrivWAW, &PrivWAR};
    for (unsigned u = 0; u < 3; u++) {
        isl_union_map **Map = Maps[u], **PrivMap = PrivMaps[u];

        *PrivMap = isl_union_map_apply_range(isl_union_map_copy(*Map), isl_union_map_copy(TC_RED));
        *PrivMap =
            isl_union_map_union(*PrivMap, isl_union_map_apply_range(isl_union_map_copy(TC_RED), isl_union_map_copy(*Map)));

        *Map = isl_union_map_union(*Map, *PrivMap);
    }
}

bool Dependences::is_parallel(isl_union_map *schedule, isl_pw_aff **min_distance_ptr) const {
    isl_union_map *deps = this->dependences(TYPE_RAW | TYPE_WAR | TYPE_WAW);
    deps = isl_union_map_apply_range(deps, isl_union_map_copy(schedule));
    deps = isl_union_map_apply_domain(deps, isl_union_map_copy(schedule));
    if (isl_union_map_is_empty(deps)) {
        isl_union_map_free(deps);
        return true;
    }

    isl_map *schedule_deps = isl_map_from_union_map(deps);
    unsigned Dimension = isl_map_dim(schedule_deps, isl_dim_out) - 1;

    for (unsigned i = 0; i < Dimension; i++)
        schedule_deps = isl_map_equate(schedule_deps, isl_dim_out, i, isl_dim_in, i);

    isl_set *Deltas = isl_map_deltas(schedule_deps);
    isl_set *Distance = isl_set_universe(isl_set_get_space(Deltas));

    // [0, ..., 0, +] - All zeros and last dimension larger than zero
    for (unsigned i = 0; i < Dimension; i++) Distance = isl_set_fix_si(Distance, isl_dim_set, i, 0);

    Distance = isl_set_lower_bound_si(Distance, isl_dim_set, Dimension, 1);
    Distance = isl_set_intersect(Distance, Deltas);

    bool IsParallel = isl_set_is_empty(Distance);
    if (IsParallel || !min_distance_ptr) {
        isl_set_free(Distance);
        return IsParallel;
    }

    Distance = isl_set_project_out(Distance, isl_dim_set, 0, Dimension);
    Distance = isl_set_coalesce(Distance);

    // This last step will compute a expression for the minimal value in the
    // distance polyhedron Distance with regards to the first (outer most)
    // dimension.
    *min_distance_ptr = isl_pw_aff_coalesce(isl_set_dim_min(Distance, 0));

    return false;
}

static bool check_validity(const Dependences *DepsObj, Scop &scop, isl_union_map *Schedule) {
    isl_union_map *Deps = DepsObj->dependences(Dependences::TYPE_RAW | Dependences::TYPE_WAW | Dependences::TYPE_WAR);

    Deps = isl_union_map_apply_domain(Deps, isl_union_map_copy(Schedule));
    Deps = isl_union_map_apply_range(Deps, isl_union_map_copy(Schedule));

    isl_space *ScheduleSpace = NULL;
    isl_union_map_foreach_map(Schedule, get_map_range_space, &ScheduleSpace);
    isl_union_map_free(Schedule);

    if (!ScheduleSpace) {
        isl_union_map_free(Deps);
        return true; /* Empty schedule considered valid/trivial */
    }

    isl_set *Zero = isl_set_universe(isl_space_copy(ScheduleSpace));
    int dims = isl_set_dim(Zero, isl_dim_set);
    for (int i = 0; i < dims; ++i) Zero = isl_set_fix_si(Zero, isl_dim_set, i, 0);

    isl_union_set *UDeltas = isl_union_map_deltas(Deps);
    isl_set *Deltas = isl_union_set_extract_set(UDeltas, ScheduleSpace);
    isl_union_set_free(UDeltas);

    isl_space *Space = isl_set_get_space(Deltas);
    isl_space *MapSpace = isl_space_map_from_set(isl_space_copy(Space));
    isl_map *NonPositive = isl_map_universe(MapSpace);

    isl_multi_pw_aff *Identity = isl_multi_pw_aff_identity_on_domain_space(Space);
    NonPositive = isl_map_lex_le_at_multi_pw_aff(NonPositive, Identity);

    NonPositive = isl_map_intersect_domain(NonPositive, Deltas);
    NonPositive = isl_map_intersect_range(NonPositive, Zero);

    bool IsEmpty = isl_map_is_empty(NonPositive);

    isl_map_free(NonPositive);
    return IsEmpty;
}

bool Dependences::is_valid(Scop &scop, const std::unordered_map<ScopStatement *, isl_map *> &new_schedule) const {
    isl_space *SpaceParams = isl_space_params_alloc(scop.ctx(), 0);
    isl_union_map *Schedule = isl_union_map_empty(SpaceParams);

    for (ScopStatement *Stmt : scop.statements()) {
        isl_map *StmtScat;

        auto Lookup = new_schedule.find(Stmt);
        if (Lookup == new_schedule.end()) {
            StmtScat = isl_map_copy(Stmt->schedule());
        } else {
            StmtScat = isl_map_copy(Lookup->second);
        }

        Schedule = isl_union_map_union(Schedule, isl_union_map_from_map(StmtScat));
    }

    return check_validity(this, scop, Schedule);
}

bool Dependences::is_valid(Scop &scop, isl_schedule *schedule) const {
    return check_validity(this, scop, isl_schedule_get_map(schedule));
}

ScopToSDFG::ScopToSDFG(Scop &scop, const Dependences &dependences, builder::StructuredSDFGBuilder &builder)
    : scop_(scop), dependences_(dependences), builder_(builder) {
    for (auto *stmt : scop_.statements()) {
        stmt_map_[stmt->name()] = stmt;
    }
}

struct NameConnectInfo {
    std::vector<std::string> names;
};

static isl_stat collect_dim_names(isl_map *map, void *user) {
    auto *info = static_cast<NameConnectInfo *>(user);
    int dim = isl_map_dim(map, isl_dim_out);
    if (dim > info->names.size()) {
        info->names.resize(dim);
    }

    isl_space *space = isl_map_get_space(map);
    isl_local_space *ls = isl_local_space_from_space(space);

    for (int i = 0; i < dim; ++i) {
        if (!info->names[i].empty()) {
            continue;
        }

        const char *name = isl_map_get_dim_name(map, isl_dim_out, i);
        if (name) {
            info->names[i] = name;
            continue;
        }

        // Try to recover from input dimensions
        int in_dim = isl_map_dim(map, isl_dim_in);
        for (int j = 0; j < in_dim; ++j) {
            isl_constraint *c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
            c = isl_constraint_set_coefficient_si(c, isl_dim_out, i, 1);
            c = isl_constraint_set_coefficient_si(c, isl_dim_in, j, -1);

            isl_basic_map *eq_bmap = isl_basic_map_from_constraint(c);
            isl_map *eq_map = isl_map_from_basic_map(eq_bmap);

            bool is_subset = isl_map_is_subset(map, eq_map);
            isl_map_free(eq_map);

            if (is_subset) {
                const char *in_name = isl_map_get_dim_name(map, isl_dim_in, j);
                if (in_name) {
                    info->names[i] = in_name;
                    break;
                }
            }
        }
    }
    isl_local_space_free(ls);
    isl_map_free(map);
    return isl_stat_ok;
}

static __isl_give isl_ast_node *after_each_for(__isl_take isl_ast_node *node, __isl_keep isl_ast_build *build, void *user) {
    auto *deps = static_cast<const Dependences *>(user);
    isl_union_map *schedule = isl_ast_build_get_schedule(build);
    bool is_parallel = deps->is_parallel(schedule);
    isl_union_map_free(schedule);

    if (is_parallel) {
        isl_id *id = isl_id_alloc(isl_ast_node_get_ctx(node), "parallel", nullptr);
        node = isl_ast_node_set_annotation(node, id);
    }
    return node;
}

structured_control_flow::ControlFlowNode &ScopToSDFG::build(analysis::AnalysisManager &analysis_manager) {
    isl_ctx *ctx = scop_.ctx();

    // 1. Create AST Builder
    isl_ast_build *ast_builder = isl_ast_build_alloc(ctx);
    ast_builder = isl_ast_build_set_after_each_for(ast_builder, &after_each_for, (void *) &dependences_);

    // 2. Generate AST from Schedule
    isl_schedule *schedule = isl_schedule_copy(scop_.schedule_tree());
    isl_ast_node *root_node = isl_ast_build_node_from_schedule(ast_builder, schedule);

    // 3. Start Traversal
    auto &scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parent_scope = scope_analysis.parent_scope(&scop_.node());
    if (!parent_scope) {
        parent_scope = &builder_.hoist_root();
    }
    auto parent_sequence = static_cast<structured_control_flow::Sequence *>(parent_scope);
    int index = parent_sequence->index(scop_.node());
    auto &target_sequence = builder_.add_sequence_before(*parent_sequence, scop_.node(), {}, scop_.node().debug_info());

    visit_node(root_node, target_sequence);
    builder_.remove_child(*parent_sequence, index + 1);

    isl_ast_node_free(root_node);
    isl_ast_build_free(ast_builder);

    return target_sequence;
}

void ScopToSDFG::visit_node(isl_ast_node *node, structured_control_flow::Sequence &scope) {
    if (!node) return;

    switch (isl_ast_node_get_type(node)) {
        case isl_ast_node_for:
            visit_for(node, scope);
            break;
        case isl_ast_node_if:
            visit_if(node, scope);
            break;
        case isl_ast_node_block:
            visit_block(node, scope);
            break;
        case isl_ast_node_mark:
            visit_mark(node, scope);
            break;
        case isl_ast_node_user:
            visit_user(node, scope);
            break;
        default: {
            throw std::runtime_error("Unsupported AST node type encountered during SCoP to SDFG conversion.");
            break;
        }
    }
}

void ScopToSDFG::visit_for(isl_ast_node *node, structured_control_flow::Sequence &scope) {
    // Extract Loop Components from ISL
    isl_ast_expr *iterator = isl_ast_node_for_get_iterator(node);
    isl_ast_expr *init = isl_ast_node_for_get_init(node);
    isl_ast_expr *cond = isl_ast_node_for_get_cond(node);
    isl_ast_expr *inc = isl_ast_node_for_get_inc(node);
    isl_ast_node *body = isl_ast_node_for_get_body(node);

    // Convert to Symbolic
    auto id = isl_ast_expr_get_id(iterator);
    symbolic::Symbol sym_iter = symbolic::symbol(isl_id_get_name(id));
    if (!builder_.subject().exists(sym_iter->get_name())) {
        builder_.add_container(sym_iter->get_name(), types::Scalar(types::PrimitiveType::Int64));
    }
    symbolic::Expression sym_init = convert_expr(init);
    symbolic::Condition sym_cond = convert_cond(cond);
    symbolic::Expression step = convert_expr(inc);
    symbolic::Expression update_expr = symbolic::add(sym_iter, step);

    // Create map for parallel loops
    bool is_parallel = false;

    isl_id *annotation = isl_ast_node_get_annotation(node);
    if (annotation) {
        if (std::string(isl_id_get_name(annotation)) == "parallel") {
            is_parallel = true;
        }
        isl_id_free(annotation);
    }

    if (is_parallel) {
        auto &loop = builder_.add_map(
            scope, sym_iter, sym_cond, sym_init, update_expr, structured_control_flow::ScheduleType_Sequential::create()
        );

        // Recurse into body
        visit_node(body, loop.root());
    } else {
        auto &loop = builder_.add_for(scope, sym_iter, sym_cond, sym_init, update_expr);

        // Recurse into body
        visit_node(body, loop.root());
    }

    isl_ast_expr_free(iterator);
    isl_ast_expr_free(init);
    isl_ast_expr_free(cond);
    isl_ast_expr_free(inc);
    isl_ast_node_free(body);
    isl_id_free(id);
}

void ScopToSDFG::visit_if(isl_ast_node *node, structured_control_flow::Sequence &scope) {
    isl_ast_expr *cond = isl_ast_node_if_get_cond(node);
    isl_ast_node *then_node = isl_ast_node_if_get_then(node);
    isl_ast_node *else_node = isl_ast_node_if_get_else(node);

    auto &if_stmt = builder_.add_if_else(scope);

    // Then Block
    symbolic::Condition sym_cond = convert_cond(cond);
    auto &then_seq = builder_.add_case(if_stmt, sym_cond);
    visit_node(then_node, then_seq);

    // Else Block (if exists)
    if (else_node) {
        auto &else_seq = builder_.add_case(if_stmt, symbolic::Not(sym_cond));
        visit_node(else_node, else_seq);
    }

    isl_ast_expr_free(cond);
    isl_ast_node_free(then_node);
    isl_ast_node_free(else_node);
}

void ScopToSDFG::visit_block(isl_ast_node *node, structured_control_flow::Sequence &scope) {
    isl_ast_node_list *list = isl_ast_node_block_get_children(node);
    int n = isl_ast_node_list_n_ast_node(list);

    // Just iterate and append to current sequence
    for (int i = 0; i < n; ++i) {
        isl_ast_node *child = isl_ast_node_list_get_ast_node(list, i);
        visit_node(child, scope);
        isl_ast_node_free(child);
    }
    isl_ast_node_list_free(list);
}

void ScopToSDFG::visit_mark(isl_ast_node *node, structured_control_flow::Sequence &scope) {
    isl_ast_node *child = isl_ast_node_mark_get_node(node);
    visit_node(child, scope);
    isl_ast_node_free(child);
}

void ScopToSDFG::visit_user(isl_ast_node *node, structured_control_flow::Sequence &scope) {
    isl_ast_expr *expr = isl_ast_node_user_get_expr(node);
    // Usually 'expr' is a call operation: "StatementName(i, j)"
    // The identifier of the operation is the statement name.

    // A more robust way to get tuple name directly from expr if it is an ID or Call:
    isl_id *id = isl_ast_expr_get_id(expr); // If it's just an ID
    if (!id && isl_ast_expr_get_type(expr) == isl_ast_expr_op) {
        // It's likely a call: S(i, j)
        // The first "arg" or the operation type might hold ID.
        // Actually, isl_ast_expr_get_op_type(expr) == isl_ast_expr_op_call
        isl_ast_expr *func = isl_ast_expr_get_op_arg(expr, 0);
        id = isl_ast_expr_get_id(func);
        isl_ast_expr_free(func);
    }

    if (id) {
        std::string name = isl_id_get_name(id);
        if (stmt_map_.count(name)) {
            ScopStatement *stmt = stmt_map_[name];

            // Get args of expr
            int n_args = isl_ast_expr_get_op_n_arg(expr);
            std::vector<symbolic::Expression> args;
            for (int i = 1; i < n_args; ++i) {
                isl_ast_expr *arg_expr = isl_ast_expr_get_op_arg(expr, i);
                symbolic::Expression arg_sym = convert_expr(arg_expr);
                args.push_back(arg_sym);
                isl_ast_expr_free(arg_expr);
            }

            // Add transition mapping stmt->iterator_i = arg_i
            assert(args.size() == stmt->iterators().size());
            Assignments assignments;
            for (size_t i = 0; i < stmt->iterators().size(); ++i) {
                assignments[stmt->iterators().at(i)] = args[i];
            }
            if (!assignments.empty()) {
                builder_.add_block(scope, assignments);
            }

            if (!stmt->expression().is_null()) {
                std::string rhs = (*stmt->writes().begin())->data();
                builder_.add_block(scope, {{symbolic::symbol(rhs), stmt->expression()}});
            } else {
                auto &new_block = builder_.add_block(scope);
                auto &new_code_node =
                    static_cast<data_flow::CodeNode &>(builder_.copy_node(new_block, *stmt->code_node()));

                // Create Access Node Mapping
                auto &old_graph = stmt->code_node()->get_parent();
                auto &new_graph = new_code_node.get_parent();
                std::unordered_map<const data_flow::DataFlowNode *, data_flow::DataFlowNode *> in_node_map;
                for (auto &iedge : old_graph.in_edges(*stmt->code_node())) {
                    if (in_node_map.find(&iedge.src()) == in_node_map.end()) {
                        in_node_map[&iedge.src()] = &builder_.copy_node(new_block, iedge.src());
                    }
                    builder_.add_memlet(
                        new_block,
                        *in_node_map[&iedge.src()],
                        iedge.src_conn(),
                        new_code_node,
                        iedge.dst_conn(),
                        iedge.subset(),
                        iedge.base_type(),
                        iedge.debug_info()
                    );
                }

                std::unordered_map<const data_flow::DataFlowNode *, data_flow::DataFlowNode *> out_node_map;
                for (auto &oedge : old_graph.out_edges(*stmt->code_node())) {
                    if (out_node_map.find(&oedge.dst()) == out_node_map.end()) {
                        out_node_map[&oedge.dst()] = &builder_.copy_node(new_block, oedge.dst());
                    }
                    builder_.add_memlet(
                        new_block,
                        new_code_node,
                        oedge.src_conn(),
                        *out_node_map[&oedge.dst()],
                        oedge.dst_conn(),
                        oedge.subset(),
                        oedge.base_type(),
                        oedge.debug_info()
                    );
                }
            }
        }
        isl_id_free(id);
    }
    isl_ast_expr_free(expr);
}

symbolic::Expression ScopToSDFG::convert_expr(isl_ast_expr *expr) {
    if (!expr) return SymEngine::null;

    isl_ast_expr_type type = isl_ast_expr_get_type(expr);
    if (type == isl_ast_expr_int) {
        isl_val *v = isl_ast_expr_get_val(expr);
        long val = isl_val_get_num_si(v);
        isl_val_free(v);
        return symbolic::integer(val);
    } else if (type == isl_ast_expr_id) {
        isl_id *id = isl_ast_expr_get_id(expr);
        std::string name = isl_id_get_name(id);
        isl_id_free(id);
        return symbolic::symbol(name);
    } else if (type == isl_ast_expr_op) {
        isl_ast_op_type op = isl_ast_expr_get_op_type(expr);
        int n_args = isl_ast_expr_get_op_n_arg(expr);

        if (n_args == 1) {
            isl_ast_expr *arg0 = isl_ast_expr_get_op_arg(expr, 0);
            symbolic::Expression res;
            if (op == isl_ast_op_minus) {
                res = symbolic::mul(symbolic::integer(-1), convert_expr(arg0));
            }
            isl_ast_expr_free(arg0);
            return res;
        }

        if (n_args == 2) {
            isl_ast_expr *arg0 = isl_ast_expr_get_op_arg(expr, 0);
            isl_ast_expr *arg1 = isl_ast_expr_get_op_arg(expr, 1);
            symbolic::Expression s0 = convert_expr(arg0);
            symbolic::Expression s1 = convert_expr(arg1);
            isl_ast_expr_free(arg0);
            isl_ast_expr_free(arg1);

            switch (op) {
                case isl_ast_op_add:
                    return symbolic::add(s0, s1);
                case isl_ast_op_sub:
                    return symbolic::sub(s0, s1);
                case isl_ast_op_mul:
                    return symbolic::mul(s0, s1);
                case isl_ast_op_div:
                case isl_ast_op_pdiv_q:
                case isl_ast_op_fdiv_q:
                    return symbolic::div(s0, s1);
                case isl_ast_op_pdiv_r:
                case isl_ast_op_zdiv_r:
                    return symbolic::mod(s0, s1);
                case isl_ast_op_min:
                    return symbolic::min(s0, s1);
                case isl_ast_op_max:
                    return symbolic::max(s0, s1);
                default:
                    break;
            }
        }
    }
    return SymEngine::null;
}

symbolic::Condition ScopToSDFG::convert_cond(isl_ast_expr *expr) {
    if (!expr) return SymEngine::null;

    if (isl_ast_expr_get_type(expr) != isl_ast_expr_op) {
        return SymEngine::null;
    }

    isl_ast_op_type op = isl_ast_expr_get_op_type(expr);
    int n_args = isl_ast_expr_get_op_n_arg(expr);

    if (n_args == 2) {
        isl_ast_expr *arg0 = isl_ast_expr_get_op_arg(expr, 0);
        isl_ast_expr *arg1 = isl_ast_expr_get_op_arg(expr, 1);

        symbolic::Condition res;

        switch (op) {
            case isl_ast_op_eq:
                res = symbolic::Eq(convert_expr(arg0), convert_expr(arg1));
                break;
            case isl_ast_op_lt:
                res = symbolic::Lt(convert_expr(arg0), convert_expr(arg1));
                break;
            case isl_ast_op_le:
                res = symbolic::Le(convert_expr(arg0), convert_expr(arg1));
                break;
            case isl_ast_op_gt:
                res = symbolic::Gt(convert_expr(arg0), convert_expr(arg1));
                break;
            case isl_ast_op_ge:
                res = symbolic::Ge(convert_expr(arg0), convert_expr(arg1));
                break;
            case isl_ast_op_and:
            case isl_ast_op_and_then:
                res = symbolic::And(convert_cond(arg0), convert_cond(arg1));
                break;
            case isl_ast_op_or:
            case isl_ast_op_or_else:
                res = symbolic::Or(convert_cond(arg0), convert_cond(arg1));
                break;
            default:
                break;
        }

        isl_ast_expr_free(arg0);
        isl_ast_expr_free(arg1);
        return res;
    }

    return SymEngine::null;
}

ScopAnalysis::ScopAnalysis(StructuredSDFG &sdfg) : Analysis(sdfg) {}

void ScopAnalysis::run(analysis::AnalysisManager &analysis_manager) {
    this->scops_.clear();
    this->dependences_.clear();

    auto &loop_analysis = analysis_manager.get<LoopAnalysis>();
    for (auto &loop : loop_analysis.loops()) {
        if (!dynamic_cast<structured_control_flow::StructuredLoop *>(loop)) {
            continue;
        }
        auto sloop = static_cast<structured_control_flow::StructuredLoop *>(loop);

        ScopBuilder builder(this->sdfg_, *sloop);
        auto scop = builder.build(analysis_manager);
        if (!scop) {
            continue;
        }
        auto dependences = std::make_unique<Dependences>(*scop);
        if (!dependences->has_valid_dependences()) {
            continue;
        }

        this->scops_[loop] = std::move(scop);
        this->dependences_[loop] = std::move(dependences);
    }
}

bool ScopAnalysis::has(const structured_control_flow::ControlFlowNode *node) const {
    return this->scops_.find(node) != this->scops_.end();
}

Scop &ScopAnalysis::scop(const structured_control_flow::ControlFlowNode *node) const { return *this->scops_.at(node); }

const Dependences &ScopAnalysis::dependences(const structured_control_flow::ControlFlowNode *node) const {
    return *this->dependences_.at(node);
}

} // namespace analysis
} // namespace sdfg
