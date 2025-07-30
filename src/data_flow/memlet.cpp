#include <sdfg/data_flow/memlet.h>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace data_flow {

Memlet::Memlet(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Edge& edge,
    DataFlowGraph& parent,
    DataFlowNode& src,
    const std::string& src_conn,
    DataFlowNode& dst,
    const std::string& dst_conn,
    const Subset& subset,
    const types::IType& base_type
)
    : Element(element_id, debug_info), edge_(edge), parent_(&parent), src_(src), dst_(dst), src_conn_(src_conn),
      dst_conn_(dst_conn), begin_subset_(subset), end_subset_(subset), base_type_(base_type.clone()) {

      };

Memlet::Memlet(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Edge& edge,
    DataFlowGraph& parent,
    DataFlowNode& src,
    const std::string& src_conn,
    DataFlowNode& dst,
    const std::string& dst_conn,
    const Subset& begin_subset,
    const Subset& end_subset,
    const types::IType& base_type
)
    : Element(element_id, debug_info), edge_(edge), parent_(&parent), src_(src), dst_(dst), src_conn_(src_conn),
      dst_conn_(dst_conn), begin_subset_(begin_subset), end_subset_(end_subset), base_type_(base_type.clone()) {

      };

void Memlet::validate(const Function& function) const {
    // Case 1: Computational Memlet (Tasklet)
    if (auto tasklet = dynamic_cast<const Tasklet*>(&this->src_)) {
        if (!dynamic_cast<const AccessNode*>(&this->dst_)) {
            throw InvalidSDFGException("Memlet: Computational memlet must have an access node destination");
        }
        if (this->dst_conn_ != "void") {
            throw InvalidSDFGException("Memlet: Computation memlets must have a void destination");
        }
        if (tasklet->output() != this->src_conn_) {
            throw InvalidSDFGException("Memlet: Computation memlet must have an output in the tasklet");
        }

        // Criterion: Must be a scalar
        auto& deref_type = types::infer_type(function, *this->base_type_, this->begin_subset_);
        if (deref_type.type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException("Memlet: Computation memlets must have a scalar destination");
        }

        return;
    } else if (auto tasklet = dynamic_cast<const Tasklet*>(&this->dst_)) {
        if (!dynamic_cast<const AccessNode*>(&this->src_)) {
            throw InvalidSDFGException("Memlet: Computation memlet must have an access node source");
        }
        if (this->src_conn_ != "void") {
            throw InvalidSDFGException("Memlet: Computation memlets must have a void source");
        }
        bool found_conn = false;
        for (auto& conn : tasklet->inputs()) {
            if (conn == this->dst_conn_) {
                found_conn = true;
                break;
            }
        }
        if (!found_conn) {
            throw InvalidSDFGException("Memlet: Computation memlet must have an input in the tasklet");
        }

        // Criterion: Must be a scalar
        auto& deref_type = types::infer_type(function, *this->base_type_, this->begin_subset_);
        if (deref_type.type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException("Memlet: Computation memlets must have a scalar source");
        }
        return;
    }

    // Case 2: Computational Memlet (LibraryNode)
    if (auto libnode = dynamic_cast<const LibraryNode*>(&this->src_)) {
        if (!dynamic_cast<const AccessNode*>(&this->dst_)) {
            throw InvalidSDFGException("Memlet: Computational memlet must have an access node destination");
        }
        if (this->dst_conn_ != "void") {
            throw InvalidSDFGException("Memlet: Computation memlets must have a void destination");
        }

        bool found_conn = false;
        for (auto& conn : libnode->outputs()) {
            if (conn == this->src_conn_) {
                found_conn = true;
                break;
            }
        }
        if (!found_conn) {
            throw InvalidSDFGException("Memlet: Computation memlet must have an output in the library node");
        }
        return;
    } else if (auto libnode = dynamic_cast<const LibraryNode*>(&this->dst_)) {
        if (!dynamic_cast<const AccessNode*>(&this->src_)) {
            throw InvalidSDFGException("Memlet: Computational memlet must have an access node source");
        }
        if (this->src_conn_ != "void") {
            throw InvalidSDFGException("Memlet: Computation memlets must have a void source");
        }

        bool found_conn = false;
        for (auto& conn : libnode->inputs()) {
            if (conn == this->dst_conn_) {
                found_conn = true;
                break;
            }
        }
        if (!found_conn) {
            throw InvalidSDFGException("Memlet: Computation memlet must have an input in the library node");
        }
        return;
    }

    // Case 3: Reference Memlet (Address Calculation)
    if (this->dst_conn_ == "ref") {
        if (this->src_conn_ != "void") {
            throw InvalidSDFGException("Memlet: Reference memlets must have a void source");
        }
        auto src_node = dynamic_cast<const AccessNode*>(&this->src_);
        if (!src_node) {
            throw InvalidSDFGException("Memlet: Reference memlets must have an access node source");
        }
        auto dst_node = dynamic_cast<const AccessNode*>(&this->dst_);
        if (!dst_node) {
            throw InvalidSDFGException("Memlet: Reference memlets must have an access node destination");
        }

        // Criterion: Reference memlets for raw addresses must not have a subset
        if (helpers::is_number(src_node->data()) || symbolic::is_nullptr(symbolic::symbol(src_node->data()))) {
            if (!this->begin_subset_.empty()) {
                throw InvalidSDFGException("Memlet: Reference memlets for raw addresses must not have a subset");
            }
        } else {
            // Criterion: Source must be a pointer
            auto src_data = src_node->data();
            auto& src_type = function.type(src_data);
            if (src_type.type_id() != types::TypeID::Pointer) {
                throw InvalidSDFGException("Memlet: Reference memlets must have a pointer source");
            }
        }

        // Criterion: Destination must be a pointer
        auto dst_data = dst_node->data();
        auto& dst_type = function.type(dst_data);
        if (dst_type.type_id() != types::TypeID::Pointer) {
            throw InvalidSDFGException("Memlet: Reference memlets must have a pointer destination");
        }

        // Criterion: Must be a contiguous memory reference
        // Throws exception if not contiguous
        auto& ref_type = types::infer_type(function, *this->base_type_, this->begin_subset_);

        return;
    }

    // Case 4: Dereference Memlet (Load/Store from/to pointer)
    if (this->src_conn_ == "deref") {
        if (this->dst_conn_ != "void") {
            throw InvalidSDFGException("Memlet: Dereference memlets must have a void destination");
        }
        auto src_node = dynamic_cast<const AccessNode*>(&this->src_);
        if (!src_node) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have an access node source");
        }
        auto dst_node = dynamic_cast<const AccessNode*>(&this->dst_);
        if (!dst_node) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have an access node destination");
        }

        // Criterion: Dereference memlets must have '0' as the only dimension
        if (this->begin_subset_.size() != 1) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have '0' as the only dimension");
        }
        if (!symbolic::eq(this->begin_subset_[0], symbolic::integer(0))) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have '0' as the only dimension");
        }

        // Criterion: Source must be a pointer
        auto src_data = src_node->data();
        auto& src_type = function.type(src_data);
        if (src_type.type_id() != types::TypeID::Pointer || helpers::is_number(src_data)) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have a pointer source");
        }

        // Criterion: Destination must be a pointer to source type
        auto dst_data = dst_node->data();
        auto& dst_type = function.type(dst_data);
        if (dst_type.type_id() != types::TypeID::Pointer) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have a pointer destination");
        }

        // Criterion: Must be typed pointer
        auto base_pointer_type = dynamic_cast<const types::Pointer*>(this->base_type_.get());
        if (!base_pointer_type) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have a typed pointer base type");
        }
        if (!base_pointer_type->has_pointee_type()) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have a pointee type");
        }

        return;
    } else if (this->dst_conn_ == "deref") {
        if (this->src_conn_ != "void") {
            throw InvalidSDFGException("Memlet: Dereference memlets must have a void source");
        }
        auto src_node = dynamic_cast<const AccessNode*>(&this->src_);
        if (!src_node) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have an access node source");
        }
        auto dst_node = dynamic_cast<const AccessNode*>(&this->dst_);
        if (!dst_node) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have an access node destination");
        }

        // Criterion: Dereference memlets must have '0' as the only dimension
        if (this->begin_subset_.size() != 1) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have '0' as the only dimension");
        }
        if (!symbolic::eq(this->begin_subset_[0], symbolic::integer(0))) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have '0' as the only dimension");
        }

        // Criterion: Source must be a pointer
        auto src_data = src_node->data();
        auto& src_type = function.type(src_data);
        if (src_type.type_id() != types::TypeID::Pointer || helpers::is_number(src_data)) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have a pointer source");
        }

        // Criterion: Destination must be a pointer to source type
        auto dst_data = dst_node->data();
        auto& dst_type = function.type(dst_data);
        if (dst_type.type_id() != types::TypeID::Pointer) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have a pointer destination");
        }

        // Criterion: Must be typed pointer
        auto base_pointer_type = dynamic_cast<const types::Pointer*>(this->base_type_.get());
        if (!base_pointer_type) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have a typed pointer base type");
        }
        if (!base_pointer_type->has_pointee_type()) {
            throw InvalidSDFGException("Memlet: Dereference memlets must have a pointee type");
        }

        return;
    }

    throw InvalidSDFGException("Memlet: Invalid memlet connection");
};

const graph::Edge Memlet::edge() const { return this->edge_; };

const DataFlowGraph& Memlet::get_parent() const { return *this->parent_; };

DataFlowGraph& Memlet::get_parent() { return *this->parent_; };

MemletType Memlet::type() const {
    if (this->dst_conn_ == "ref") {
        return Reference;
    } else if (this->dst_conn_ == "deref") {
        return Dereference_Src;
    } else if (this->src_conn_ == "deref") {
        return Dereference_Dst;
    } else {
        return Computational;
    }
}

const DataFlowNode& Memlet::src() const { return this->src_; };

DataFlowNode& Memlet::src() { return this->src_; };

const DataFlowNode& Memlet::dst() const { return this->dst_; };

DataFlowNode& Memlet::dst() { return this->dst_; };

const std::string& Memlet::src_conn() const { return this->src_conn_; };

const std::string& Memlet::dst_conn() const { return this->dst_conn_; };

const Subset Memlet::subset() const { return this->begin_subset_; };

void Memlet::set_subset(const Subset& subset) {
    this->begin_subset_ = subset;
    this->end_subset_ = subset;
};

const Subset Memlet::begin_subset() const { return this->begin_subset_; };

const Subset Memlet::end_subset() const { return this->end_subset_; };

void Memlet::set_subset(const Subset& begin_subset, const Subset& end_subset) {
    this->begin_subset_ = begin_subset;
    this->end_subset_ = end_subset;
};

bool Memlet::has_range() const {
    for (size_t i = 0; i < this->begin_subset_.size(); i++) {
        if (!symbolic::eq(this->begin_subset_[i], this->end_subset_[i])) {
            return true;
        }
    }
    return false;
};

const types::IType& Memlet::base_type() const { return *this->base_type_; };

void Memlet::set_base_type(const types::IType& base_type) { this->base_type_ = base_type.clone(); };

const types::IType& Memlet::result_type(const Function& function) const {
    return types::infer_type(function, *this->base_type_, this->begin_subset_);
};

std::unique_ptr<Memlet> Memlet::clone(
    size_t element_id, const graph::Edge& edge, DataFlowGraph& parent, DataFlowNode& src, DataFlowNode& dst
) const {
    return std::unique_ptr<Memlet>(new Memlet(
        element_id,
        this->debug_info_,
        edge,
        parent,
        src,
        this->src_conn_,
        dst,
        this->dst_conn_,
        this->begin_subset_,
        this->end_subset_,
        *this->base_type_
    ));
};

void Memlet::replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) {
    for (auto& dim : this->begin_subset_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
    for (auto& dim : this->end_subset_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
};

} // namespace data_flow
} // namespace sdfg
