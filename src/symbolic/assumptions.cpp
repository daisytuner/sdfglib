#include "sdfg/symbolic/assumptions.h"

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

Assumption::Assumption()
    : symbol_(symbolic::symbol("")),
      lower_bound_(symbolic::infty(-1)),
      upper_bound_(symbolic::infty(1)) {

      };

Assumption::Assumption(const Symbol& symbol)
    : symbol_(symbol), lower_bound_(symbolic::infty(-1)), upper_bound_(symbolic::infty(1)) {

      };

Assumption::Assumption(const Assumption& a)
    : symbol_(a.symbol_), lower_bound_(a.lower_bound_), upper_bound_(a.upper_bound_) {

      };

Assumption& Assumption::operator=(const Assumption& a) {
    symbol_ = a.symbol_;
    lower_bound_ = a.lower_bound_;
    upper_bound_ = a.upper_bound_;
    map_ = a.map_;
    return *this;
};

const Symbol& Assumption::symbol() const { return symbol_; };

const Expression& Assumption::lower_bound() const { return lower_bound_; };

void Assumption::lower_bound(const Expression& lower_bound) {
    lower_bound_ = symbolic::simplify(lower_bound);
};

const Expression& Assumption::upper_bound() const { return upper_bound_; };

void Assumption::upper_bound(const Expression& upper_bound) {
    upper_bound_ = symbolic::simplify(upper_bound);
};

const Integer Assumption::integer_value() const {
    if (eq(upper_bound_, lower_bound_)) {
        if (is_a<SymEngine::Integer>(*upper_bound_)) {
            return rcp_static_cast<const SymEngine::Integer>(upper_bound_);
        }
    }
    return SymEngine::null;
};

const Expression& Assumption::map() const { return map_; };

void Assumption::map(const Expression& map) {
    if (map == SymEngine::null) {
        map_ = map;
        return;
    }
    map_ = symbolic::simplify(map);
};

bool Assumption::is_positive() const {
    if (SymEngine::is_a<SymEngine::Integer>(*lower_bound_)) {
        auto lower_bound = SymEngine::rcp_static_cast<const SymEngine::Integer>(lower_bound_);
        if (lower_bound->as_int() > 0) {
            return true;
        }
    }
    return false;
};

bool Assumption::is_negative() const {
    if (SymEngine::is_a<SymEngine::Integer>(*upper_bound_)) {
        auto upper_bound = SymEngine::rcp_static_cast<const SymEngine::Integer>(upper_bound_);
        if (upper_bound->as_int() < 0) {
            return true;
        }
    }
    return false;
};

bool Assumption::is_nonnegative() const {
    if (SymEngine::is_a<SymEngine::Integer>(*lower_bound_)) {
        auto upper_bound = SymEngine::rcp_static_cast<const SymEngine::Integer>(lower_bound_);
        if (upper_bound->as_int() >= 0) {
            return true;
        }
    }
    return false;
};

bool Assumption::is_nonpositive() const {
    if (SymEngine::is_a<SymEngine::Integer>(*upper_bound_)) {
        auto upper_bound = SymEngine::rcp_static_cast<const SymEngine::Integer>(upper_bound_);
        if (upper_bound->as_int() <= 0) {
            return true;
        }
    }
    return false;
};

Assumption Assumption::create(const symbolic::Symbol& symbol, const types::IType& type) {
    if (auto scalar_type = dynamic_cast<const types::Scalar*>(&type)) {
        auto assum = Assumption(symbol);
        assum.map(SymEngine::null);

        types::PrimitiveType primitive_type = scalar_type->primitive_type();
        switch (primitive_type) {
            case types::PrimitiveType::Bool: {
                assum.lower_bound(symbolic::zero());
                assum.upper_bound(symbolic::one());
                break;
            }
            case types::PrimitiveType::UInt8: {
                assum.lower_bound(symbolic::integer(std::numeric_limits<uint8_t>::min()));
                assum.upper_bound(symbolic::integer(std::numeric_limits<uint8_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt16: {
                assum.lower_bound(symbolic::integer(std::numeric_limits<uint16_t>::min()));
                assum.upper_bound(symbolic::integer(std::numeric_limits<uint16_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt32: {
                assum.lower_bound(symbolic::integer(std::numeric_limits<uint32_t>::min()));
                assum.upper_bound(symbolic::integer(std::numeric_limits<uint32_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt64: {
                assum.lower_bound(symbolic::integer(std::numeric_limits<uint64_t>::min()));
                assum.upper_bound(symbolic::infty(1));
                break;
            }
            case types::PrimitiveType::Int8: {
                assum.lower_bound(symbolic::integer(std::numeric_limits<int8_t>::min()));
                assum.upper_bound(symbolic::integer(std::numeric_limits<int8_t>::max()));
                break;
            }
            case types::PrimitiveType::Int16: {
                assum.lower_bound(symbolic::integer(std::numeric_limits<int16_t>::min()));
                assum.upper_bound(symbolic::integer(std::numeric_limits<int16_t>::max()));
                break;
            }
            case types::PrimitiveType::Int32: {
                assum.lower_bound(symbolic::integer(std::numeric_limits<int32_t>::min()));
                assum.upper_bound(symbolic::integer(std::numeric_limits<int32_t>::max()));
                break;
            }
            case types::PrimitiveType::Int64: {
                assum.lower_bound(symbolic::infty(-1));
                assum.upper_bound(symbolic::infty(1));
                break;
            }
            default: {
                throw std::runtime_error("Unsupported type");
            }
        };
        return std::move(assum);
    } else {
        throw std::runtime_error("Unsupported type");
    }
};

void upper_bounds(const symbolic::Symbol& sym, const Assumptions& assumptions,
                  symbolic::SymbolicSet& ubs, symbolic::SymbolicSet& visited) {
    if (visited.find(sym) != visited.end()) {
        return;
    }
    visited.insert(sym);

    auto ub = assumptions.at(sym).upper_bound();
    if (SymEngine::is_a<SymEngine::Integer>(*ub)) {
        ubs.insert(ub);
    } else if (SymEngine::is_a<SymEngine::Symbol>(*ub)) {
        auto ub_sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(ub);
        ubs.insert(ub_sym);

        upper_bounds(ub_sym, assumptions, ubs, visited);
    } else if (SymEngine::is_a<SymEngine::Min>(*ub)) {
        auto ub_min = SymEngine::rcp_static_cast<const SymEngine::Min>(ub);
        for (auto& arg : ub_min->get_args()) {
            if (SymEngine::is_a<SymEngine::Integer>(*ub)) {
                ubs.insert(ub);
            } else if (SymEngine::is_a<SymEngine::Symbol>(*ub)) {
                auto ub_sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(ub);
                ubs.insert(ub_sym);

                upper_bounds(ub_sym, assumptions, ubs, visited);
            } else {
                ubs.insert(arg);
            }
        }
    } else {
        ubs.insert(ub);
    }
};

void upper_bounds(const symbolic::Symbol& sym, const Assumptions& assumptions,
                  symbolic::SymbolicSet& ubs) {
    symbolic::SymbolicSet visited;
    upper_bounds(sym, assumptions, ubs, visited);
};

void lower_bounds(const symbolic::Symbol& sym, const Assumptions& assumptions,
                  symbolic::SymbolicSet& lbs, symbolic::SymbolicSet& visited) {
    if (visited.find(sym) != visited.end()) {
        return;
    }
    visited.insert(sym);

    auto lb = assumptions.at(sym).lower_bound();
    if (SymEngine::is_a<SymEngine::Integer>(*lb)) {
        lbs.insert(lb);
    } else if (SymEngine::is_a<SymEngine::Symbol>(*lb)) {
        auto lb_sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(lb);
        lbs.insert(lb_sym);

        lower_bounds(lb_sym, assumptions, lbs, visited);
    } else if (SymEngine::is_a<SymEngine::Max>(*lb)) {
        auto lb_max = SymEngine::rcp_static_cast<const SymEngine::Max>(lb);
        for (auto& arg : lb_max->get_args()) {
            if (SymEngine::is_a<SymEngine::Integer>(*lb)) {
                lbs.insert(lb);
            } else if (SymEngine::is_a<SymEngine::Symbol>(*lb)) {
                auto lb_sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(lb);
                lbs.insert(lb_sym);

                lower_bounds(lb_sym, assumptions, lbs, visited);
            } else {
                lbs.insert(arg);
            }
        }
    } else {
        lbs.insert(lb);
    }
};

void lower_bounds(const symbolic::Symbol& sym, const Assumptions& assumptions,
                  symbolic::SymbolicSet& lbs) {
    symbolic::SymbolicSet visited;
    lower_bounds(sym, assumptions, lbs, visited);
};

}  // namespace symbolic
}  // namespace sdfg
