#include "sdfg/structured_control_flow/for_each.h"

#include "sdfg/function.h"

namespace sdfg {
namespace structured_control_flow {

ForEach::
    ForEach(size_t element_id,
        const DebugInfo& debug_info,
        symbolic::Symbol iterator,
        symbolic::Symbol end)
    : ControlFlowNode(element_id, debug_info), iterator_(iterator), end_(end) {
        this->root_ = std::unique_ptr<Sequence>(new Sequence(++element_id, debug_info));
    }

void ForEach::validate(const Function& function) const { 
    root_->validate(function);

    if (iterator_.is_null()) {
        throw InvalidSDFGException("ForEach node has a null iterator.");
    }
    if (end_.is_null()) {
        throw InvalidSDFGException("ForEach node has a null end.");
    }

    // Criterion: Iterator must be pointer
    auto& iterator_type = function.type(iterator_->get_name());
    if (iterator_type.type_id() != types::TypeID::Pointer) {
        throw InvalidSDFGException("ForEach iterator must be of pointer type.");
    }

    // Criterion: End must be pointer
    if (!symbolic::eq(end_, symbolic::__nullptr__())) {
        auto& end_type = function.type(end_->get_name());
        if (end_type.type_id() != types::TypeID::Pointer) {
            throw InvalidSDFGException("ForEach end must be of pointer type.");
        }
    }
};

const symbolic::Symbol ForEach::iterator() const {
    return iterator_;
}

const symbolic::Symbol ForEach::end() const {
    return end_;
}

Sequence& ForEach::root() const {
    return *root_;
}

void ForEach::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    root_->replace(old_expression, new_expression);

    if (symbolic::eq(iterator_, old_expression)) {
        iterator_ = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(new_expression);
    }

    if (symbolic::eq(end_, old_expression)) {
        end_ = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(new_expression);
    }
}

} // namespace structured_control_flow
} // namespace sdfg
