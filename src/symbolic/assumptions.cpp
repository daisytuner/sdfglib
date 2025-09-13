#include "sdfg/symbolic/assumptions.h"

#include "sdfg/types/scalar.h"

namespace sdfg {
namespace symbolic {

Assumption::Assumption()
    : symbol_(symbolic::symbol("")), lower_bound_(SymEngine::NegInf), upper_bound_(SymEngine::Inf), constant_(false),
      map_(SymEngine::null) {

      };

Assumption::Assumption(const Symbol symbol)
    : symbol_(symbol), lower_bound_(SymEngine::NegInf), upper_bound_(SymEngine::Inf), constant_(false),
      map_(SymEngine::null) {

      };

Assumption::Assumption(const Assumption& a)
    : symbol_(a.symbol_), lower_bound_(a.lower_bound_), upper_bound_(a.upper_bound_), constant_(a.constant_),
      map_(a.map_) {

      };

Assumption& Assumption::operator=(const Assumption& a) {
    symbol_ = a.symbol_;
    lower_bound_ = a.lower_bound_;
    upper_bound_ = a.upper_bound_;
    constant_ = a.constant_;
    map_ = a.map_;
    return *this;
};

const Symbol Assumption::symbol() const { return symbol_; };

const Expression Assumption::lower_bound() const { return lower_bound_; };

void Assumption::lower_bound(const Expression lower_bound) { lower_bound_ = lower_bound; };

const Expression Assumption::upper_bound() const { return upper_bound_; };

void Assumption::upper_bound(const Expression upper_bound) { upper_bound_ = upper_bound; };

bool Assumption::constant() const { return constant_; };

void Assumption::constant(bool constant) { constant_ = constant; };

const Expression Assumption::map() const { return map_; };

void Assumption::map(const Expression map) { map_ = map; };

Assumption Assumption::create(const Symbol symbol, const types::IType& type) {
    if (auto scalar_type = dynamic_cast<const types::Scalar*>(&type)) {
        auto assum = Assumption(symbol);

        types::PrimitiveType primitive_type = scalar_type->primitive_type();
        switch (primitive_type) {
            case types::PrimitiveType::Bool: {
                assum.lower_bound(zero());
                assum.upper_bound(one());
                break;
            }
            case types::PrimitiveType::UInt8: {
                assum.lower_bound(integer(std::numeric_limits<uint8_t>::min()));
                assum.upper_bound(integer(std::numeric_limits<uint8_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt16: {
                assum.lower_bound(integer(std::numeric_limits<uint16_t>::min()));
                assum.upper_bound(integer(std::numeric_limits<uint16_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt32: {
                assum.lower_bound(integer(std::numeric_limits<uint32_t>::min()));
                assum.upper_bound(integer(std::numeric_limits<uint32_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt64: {
                assum.lower_bound(integer(std::numeric_limits<uint64_t>::min()));
                assum.upper_bound(SymEngine::Inf);
                break;
            }
            case types::PrimitiveType::UInt128: {
                assum.lower_bound(integer(0));
                assum.upper_bound(SymEngine::Inf);
                break;
            }
            case types::PrimitiveType::Int8: {
                assum.lower_bound(integer(std::numeric_limits<int8_t>::min()));
                assum.upper_bound(integer(std::numeric_limits<int8_t>::max()));
                break;
            }
            case types::PrimitiveType::Int16: {
                assum.lower_bound(integer(std::numeric_limits<int16_t>::min()));
                assum.upper_bound(integer(std::numeric_limits<int16_t>::max()));
                break;
            }
            case types::PrimitiveType::Int32: {
                assum.lower_bound(integer(std::numeric_limits<int32_t>::min()));
                assum.upper_bound(integer(std::numeric_limits<int32_t>::max()));
                break;
            }
            case types::PrimitiveType::Int64: {
                assum.lower_bound(SymEngine::NegInf);
                assum.upper_bound(SymEngine::Inf);
                break;
            }
            case types::PrimitiveType::Int128: {
                assum.lower_bound(SymEngine::NegInf);
                assum.upper_bound(SymEngine::Inf);
                break;
            }
            default: {
                throw std::runtime_error("Unsupported type");
            }
        };
        return assum;
    } else {
        throw std::runtime_error("Unsupported type");
    }
};

void upper_bounds(const Symbol sym, const Assumptions& assumptions, ExpressionSet& ubs, ExpressionSet& visited) {
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

void upper_bounds(const Symbol sym, const Assumptions& assumptions, ExpressionSet& ubs) {
    ExpressionSet visited;
    upper_bounds(sym, assumptions, ubs, visited);
};

void lower_bounds(const Symbol sym, const Assumptions& assumptions, ExpressionSet& lbs, ExpressionSet& visited) {
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

void lower_bounds(const Symbol sym, const Assumptions& assumptions, ExpressionSet& lbs) {
    ExpressionSet visited;
    lower_bounds(sym, assumptions, lbs, visited);
};

} // namespace symbolic
} // namespace sdfg
