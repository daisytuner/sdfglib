#include "sdfg/structured_control_flow/for_each.h"

#include "sdfg/function.h"

namespace sdfg {
namespace structured_control_flow {

ForEach::
    ForEach(size_t element_id,
        const DebugInfo& debug_info,
        symbolic::Symbol iterator,
        symbolic::Symbol end,
        symbolic::Symbol update,
        symbolic::Symbol init
    )
    : ControlFlowNode(element_id, debug_info), iterator_(iterator), update_(update), end_(end), init_(init) {
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
    if (update_.is_null()) {
        throw InvalidSDFGException("ForEach node has a null update.");
    }
    if (!init_.is_null() && symbolic::eq(init_, end_)) {
        throw InvalidSDFGException("ForEach node has identical init and end symbols.");
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

    // Criterion: Update must be pointer
    auto& update_type = function.type(update_->get_name());
    if (update_type.type_id() != types::TypeID::Pointer) {
        throw InvalidSDFGException("ForEach update must be of pointer type.");
    }

    // Criterion: Init must be pointer
    if (!init_.is_null()) {
        auto& init_type = function.type(init_->get_name());
        if (init_type.type_id() != types::TypeID::Pointer) {
            throw InvalidSDFGException("ForEach init must be of pointer type.");
        }
    }
};

const symbolic::Symbol ForEach::iterator() const {
    return iterator_;
}

const symbolic::Symbol ForEach::end() const {
    return end_;
}

const symbolic::Symbol ForEach::update() const {
    return update_;
}

const symbolic::Symbol ForEach::init() const {
    return init_;
}

bool ForEach::has_init() const {
    return !init_.is_null();
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

    if (symbolic::eq(update_, old_expression)) {
        update_ = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(new_expression);
    }

    if (symbolic::eq(init_, old_expression)) {
        init_ = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(new_expression);
    }
}

} // namespace structured_control_flow
} // namespace sdfg
