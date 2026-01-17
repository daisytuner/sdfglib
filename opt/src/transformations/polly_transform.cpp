//===- ScheduleOptimizer.cpp - Calculate an optimized schedule ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass generates an entirely new schedule tree from the data dependences
// and iteration domains. The new schedule tree is computed in two steps:
//
// 1) The isl scheduling optimizer is run
//
// The isl scheduling optimizer creates a new schedule tree that maximizes
// parallelism and tileability and minimizes data-dependence distances. The
// algorithm used is a modified version of the ``Pluto'' algorithm:
//
//   U. Bondhugula, A. Hartono, J. Ramanujam, and P. Sadayappan.
//   A Practical Automatic Polyhedral Parallelizer and Locality Optimizer.
//   In Proceedings of the 2008 ACM SIGPLAN Conference On Programming Language
//   Design and Implementation, PLDI ’08, pages 101–113. ACM, 2008.
//
// 2) A set of post-scheduling transformations is applied on the schedule tree.
//
// These optimizations include:
//
//  - Tiling of the innermost tilable bands
//  - Prevectorization - The choice of a possible outer loop that is strip-mined
//                       to the innermost level to enable inner-loop
//                       vectorization.
//  - Some optimizations for spatial locality are also planned.
//
// For a detailed description of the schedule tree itself please see section 6
// of:
//
// Polyhedral AST generation is more than scanning polyhedra
// Tobias Grosser, Sven Verdoolaege, Albert Cohen
// ACM Transactions on Programming Languages and Systems (TOPLAS),
// 37(4), July 2015
// http://www.grosser.es/#pub-polyhedral-AST-generation
//
// This publication also contains a detailed discussion of the different options
// for polyhedral loop unrolling, full/partial tile separation and other uses
// of the schedule tree.
//
//===----------------------------------------------------------------------===//
//
// This file is based on the original source files which were modified for sdfglib.
//
//===----------------------------------------------------------------------===//
#include "sdfg/transformations/polly_transform.h"

#include <isl/constraint.h>
#include <isl/id.h>
#include <isl/map.h>
#include <isl/schedule_node.h>
#include <isl/space.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>

namespace sdfg {
namespace transformations {

static bool isSimpleInnermostBand(isl_schedule_node* node) {
    assert(isl_schedule_node_get_type(node) == isl_schedule_node_band);
    assert(isl_schedule_node_n_children(node) == 1);

    isl_schedule_node* child = isl_schedule_node_get_child(node, 0);
    enum isl_schedule_node_type child_type = isl_schedule_node_get_type(child);

    if (child_type == isl_schedule_node_leaf) {
        isl_schedule_node_free(child);
        return true;
    }

    if (child_type != isl_schedule_node_sequence) {
        isl_schedule_node_free(child);
        return false;
    }

    int nc = isl_schedule_node_n_children(child);
    for (int c = 0; c < nc; ++c) {
        isl_schedule_node* seq_child = isl_schedule_node_get_child(child, c);
        if (isl_schedule_node_get_type(seq_child) != isl_schedule_node_filter) {
            isl_schedule_node_free(seq_child);
            isl_schedule_node_free(child);
            return false;
        }

        isl_schedule_node* filter_child = isl_schedule_node_get_child(seq_child, 0);
        bool is_leaf = (isl_schedule_node_get_type(filter_child) == isl_schedule_node_leaf);
        isl_schedule_node_free(filter_child);
        isl_schedule_node_free(seq_child);

        if (!is_leaf) {
            isl_schedule_node_free(child);
            return false;
        }
    }

    isl_schedule_node_free(child);
    return true;
}

static bool isOneTimeParentBandNode(isl_schedule_node* node) {
    if (isl_schedule_node_get_type(node) != isl_schedule_node_band) return false;

    if (isl_schedule_node_n_children(node) != 1) return false;

    return true;
}

static bool isTileableBandNode(isl_schedule_node* node) {
    if (!isOneTimeParentBandNode(node)) return false;

    if (!isl_schedule_node_band_get_permutable(node)) return false;

    isl_space* space = isl_schedule_node_band_get_space(node);
    int dim = isl_space_dim(space, isl_dim_set);
    isl_space_free(space);

    if (dim <= 1) return false;

    return isSimpleInnermostBand(node);
}

static isl_schedule_node* tile_node(isl_ctx* ctx, isl_schedule_node* node, const char* identifier, int default_tile_size) {
    isl_space* space = isl_schedule_node_band_get_space(node);
    int dims = isl_space_dim(space, isl_dim_set);

    isl_multi_val* sizes = isl_multi_val_zero(space);
    std::string identifierString(identifier);
    for (int i = 0; i < dims; ++i) {
        sizes = isl_multi_val_set_val(sizes, i, isl_val_int_from_si(ctx, default_tile_size));
    }

    std::string tileLoopMarkerStr = identifierString + " - Tiles";
    isl_id* tileLoopMarker = isl_id_alloc(ctx, tileLoopMarkerStr.c_str(), nullptr);
    node = isl_schedule_node_insert_mark(node, tileLoopMarker);
    node = isl_schedule_node_child(node, 0);
    node = isl_schedule_node_band_tile(node, sizes);
    node = isl_schedule_node_child(node, 0);

    std::string pointLoopMarkerStr = identifierString + " - Points";
    isl_id* pointLoopMarker = isl_id_alloc(ctx, pointLoopMarkerStr.c_str(), nullptr);

    node = isl_schedule_node_insert_mark(node, pointLoopMarker);
    return isl_schedule_node_child(node, 0);
}

static isl_schedule_node* optimize_band(isl_schedule_node* node, void* User) {
    if (!isTileableBandNode(node)) {
        return node;
    }

    isl_ctx* ctx = isl_schedule_node_get_ctx(node);
    // First level tiling
    node = tile_node(ctx, node, "1st level tiling", 32);

    // Second level tiling
    node = tile_node(ctx, node, "2nd level tiling", 16);

    return node;
}

PollyTransform::PollyTransform(structured_control_flow::StructuredLoop& loop, bool tile)
    : loop_(loop), tile_(tile), scop_(nullptr), dependences_(nullptr) {};

std::string PollyTransform::name() const { return "PollyTransform"; };

bool PollyTransform::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    analysis::ScopBuilder scop_builder(builder.subject(), loop_);
    this->scop_ = scop_builder.build(analysis_manager);
    if (!this->scop_) {
        return false;
    }
    this->dependences_ = std::make_unique<analysis::Dependences>(*this->scop_);
    if (!this->dependences_->has_valid_dependences()) {
        this->dependences_ = nullptr;
        this->scop_ = nullptr;
        return false;
    }

    return true;
};

void PollyTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    assert(
        this->scop_ != nullptr && this->dependences_ != nullptr &&
        "PollyTransform::apply called without successful can_be_applied"
    );

    int validity_kinds = analysis::Dependences::TYPE_RAW | analysis::Dependences::TYPE_WAR |
                         analysis::Dependences::TYPE_WAW;
    int proximity_kinds = analysis::Dependences::TYPE_RAW | analysis::Dependences::TYPE_WAR |
                          analysis::Dependences::TYPE_WAW;
    isl_union_map* validity = this->dependences_->dependences(validity_kinds);
    isl_union_map* proximity = this->dependences_->dependences(proximity_kinds);

    isl_union_set* domain = this->scop_->domains();
    validity = isl_union_map_gist_domain(validity, isl_union_set_copy(domain));
    validity = isl_union_map_gist_range(validity, isl_union_set_copy(domain));
    proximity = isl_union_map_gist_domain(proximity, isl_union_set_copy(domain));
    proximity = isl_union_map_gist_range(proximity, isl_union_set_copy(domain));

    // Add spatial proximity
    isl_union_map* spatial_proximity = isl_union_map_empty(isl_union_map_get_space(proximity));

    for (auto* stmt : this->scop_->statements()) {
        for (auto* access : stmt->accesses()) {
            isl_map* rel = access->relation();
            if (isl_map_dim(rel, isl_dim_out) == 0) {
                continue;
            }

            // We create a map that maps each element to the next element in the last dimension
            isl_space* array_space = isl_space_range(isl_map_get_space(rel));
            isl_map* adj = isl_map_universe(isl_space_map_from_set(isl_space_copy(array_space)));

            int n_dim = isl_map_dim(adj, isl_dim_out);
            for (int i = 0; i < n_dim - 1; ++i) {
                adj = isl_map_equate(adj, isl_dim_in, i, isl_dim_out, i);
            }

            isl_local_space* ls = isl_local_space_from_space(isl_map_get_space(adj));
            isl_constraint* c = isl_equality_alloc(ls);
            c = isl_constraint_set_coefficient_si(c, isl_dim_out, n_dim - 1, 1);
            c = isl_constraint_set_coefficient_si(c, isl_dim_in, n_dim - 1, -1);
            c = isl_constraint_set_constant_si(c, -1);
            adj = isl_map_add_constraint(adj, c);

            // S -> Array
            isl_map* rel_map = isl_map_copy(rel);
            // S_next -> Array_next = Array + 1
            // We want S -> S_next such that Acc(S_next) = Acc(S) + 1
            // S -> Acc(S) -> Acc(S)+1 -> S_next
            // Proximity = Rel . Adj . Rel^-1

            isl_map* map = isl_map_apply_range(rel_map, adj);
            map = isl_map_apply_range(map, isl_map_reverse(isl_map_copy(rel)));

            spatial_proximity = isl_union_map_add_map(spatial_proximity, map);

            isl_space_free(array_space);
        }
    }

    proximity = isl_union_map_union(proximity, spatial_proximity);

    int isl_maximize_bands = 1;
    int isl_outer_coincidence = 0;

    isl_options_set_schedule_outer_coincidence(scop_->ctx(), isl_outer_coincidence);
    isl_options_set_schedule_maximize_band_depth(scop_->ctx(), isl_maximize_bands);
    isl_options_set_schedule_max_constant_term(scop_->ctx(), 20);
    isl_options_set_schedule_max_coefficient(scop_->ctx(), 20);
    isl_options_set_tile_scale_tile_loops(scop_->ctx(), 0);

    isl_schedule_constraints* SC = isl_schedule_constraints_on_domain(isl_union_set_copy(domain));
    SC = isl_schedule_constraints_set_proximity(SC, isl_union_map_copy(proximity));
    SC = isl_schedule_constraints_set_validity(SC, isl_union_map_copy(validity));
    SC = isl_schedule_constraints_set_coincidence(SC, isl_union_map_copy(validity));
    isl_schedule* S = isl_schedule_constraints_compute_schedule(SC);

    if (this->tile_) {
        isl_schedule_node* root = isl_schedule_get_root(S);
        root = isl_schedule_node_map_descendant_bottom_up(root, optimize_band, nullptr);
        isl_schedule_free(S);

        S = isl_schedule_node_get_schedule(root);
        isl_schedule_node_free(root);
    }

    scop_->set_schedule_tree(S);

    isl_union_map_free(validity);
    isl_union_map_free(proximity);
    isl_union_set_free(domain);

    DEBUG_PRINTLN("PollyTransform:" << std::endl << scop_->ast());

    auto& sdfg = builder.subject();
    analysis::ScopToSDFG converter(*scop_, *dependences_, builder);
    converter.build(analysis_manager);

    this->applied_ = true;
    this->dependences_ = nullptr;
    this->scop_ = nullptr;
};

void PollyTransform::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::For*>(&loop_)) {
        loop_type = "for";
    } else if (dynamic_cast<structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    } else {
        throw InvalidSDFGException("Unsupported loop type for serialization of loop: " + loop_.indvar()->get_name());
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->loop_.element_id()}, {"type", loop_type}}}};
    j["parameters"] = {{"tile", this->tile_}};
};

PollyTransform PollyTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    bool tile = desc["parameters"]["tile"].get<bool>();

    return PollyTransform(*loop, tile);
};

} // namespace transformations
} // namespace sdfg
