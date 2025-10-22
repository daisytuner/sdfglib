#include "sdfg/symbolic/assumptions.h"

#include "sdfg/types/scalar.h"

namespace sdfg {
namespace symbolic {

Assumption::Assumption()
    : symbol_(symbolic::symbol("")), lower_bound_deprecated_(SymEngine::NegInf),
      upper_bound_deprecated_(SymEngine::Inf), lower_bounds_(), upper_bounds_(), tight_lower_bound_(SymEngine::null),
      tight_upper_bound_(SymEngine::null), constant_(false), map_(SymEngine::null) {

      };

Assumption::Assumption(const Symbol symbol)
    : symbol_(symbol), lower_bound_deprecated_(SymEngine::NegInf), upper_bound_deprecated_(SymEngine::Inf),
      lower_bounds_(), upper_bounds_(), tight_lower_bound_(SymEngine::null), tight_upper_bound_(SymEngine::null),
      constant_(false), map_(SymEngine::null) {

      };

Assumption::Assumption(const Assumption& a)
    : symbol_(a.symbol_), lower_bound_deprecated_(a.lower_bound_deprecated_),
      upper_bound_deprecated_(a.upper_bound_deprecated_), lower_bounds_(a.lower_bounds_),
      upper_bounds_(a.upper_bounds_), tight_lower_bound_(a.tight_lower_bound_),
      tight_upper_bound_(a.tight_upper_bound_), constant_(a.constant_), map_(a.map_) {

      };

Assumption& Assumption::operator=(const Assumption& a) {
    this->symbol_ = a.symbol_;
    this->lower_bound_deprecated_ = a.lower_bound_deprecated_;
    this->upper_bound_deprecated_ = a.upper_bound_deprecated_;
    this->lower_bounds_ = a.lower_bounds_;
    this->upper_bounds_ = a.upper_bounds_;
    this->tight_lower_bound_ = a.tight_lower_bound_;
    this->tight_upper_bound_ = a.tight_upper_bound_;
    this->constant_ = a.constant_;
    this->map_ = a.map_;
    return *this;
};

const Symbol Assumption::symbol() const { return this->symbol_; };

const Expression Assumption::lower_bound_deprecated() const { return this->lower_bound_deprecated_; }

void Assumption::lower_bound_deprecated(const Expression lower_bound) { this->lower_bound_deprecated_ = lower_bound; }

const Expression Assumption::upper_bound_deprecated() const { return this->upper_bound_deprecated_; }

void Assumption::upper_bound_deprecated(const Expression upper_bound) { this->upper_bound_deprecated_ = upper_bound; }

const Expression Assumption::lower_bound() const {
    Expression lb;
    if (this->lower_bounds_.empty()) {
        lb = SymEngine::NegInf;
    } else {
        lb = SymEngine::max(std::vector<Expression>(this->lower_bounds_.begin(), this->lower_bounds_.end()));
    }
    if (this->tight_lower_bound_.is_null()) {
        return lb;
    } else if (eq(lb, SymEngine::NegInf)) {
        return this->tight_lower_bound_;
    } else {
        return max(this->tight_lower_bound_, lb);
    }
}

const ExpressionSet& Assumption::lower_bounds() const { return this->lower_bounds_; }

void Assumption::add_lower_bound(const Expression lb) { this->lower_bounds_.insert(lb); }

bool Assumption::contains_lower_bound(const Expression lb) { return this->lower_bounds_.contains(lb); }

bool Assumption::remove_lower_bound(const Expression lb) { return this->lower_bounds_.erase(lb) > 0; }

const Expression Assumption::upper_bound() const {
    Expression ub;
    if (this->upper_bounds_.empty()) {
        ub = SymEngine::Inf;
    } else {
        ub = SymEngine::min(std::vector<Expression>(this->upper_bounds_.begin(), this->upper_bounds_.end()));
    }
    if (this->tight_upper_bound_.is_null()) {
        return ub;
    } else if (eq(ub, SymEngine::Inf)) {
        return this->tight_upper_bound_;
    } else {
        return min(this->tight_upper_bound_, ub);
    }
}

const ExpressionSet& Assumption::upper_bounds() const { return this->upper_bounds_; }

void Assumption::add_upper_bound(const Expression ub) { this->upper_bounds_.insert(ub); }

bool Assumption::contains_upper_bound(const Expression ub) { return this->upper_bounds_.contains(ub); }

bool Assumption::remove_upper_bound(const Expression ub) { return this->upper_bounds_.erase(ub) > 0; }

const Expression Assumption::tight_lower_bound() const { return this->tight_lower_bound_; }

void Assumption::tight_lower_bound(const Expression tight_lb) { this->tight_lower_bound_ = tight_lb; }

const Expression Assumption::tight_upper_bound() const { return this->tight_upper_bound_; }

void Assumption::tight_upper_bound(const Expression tight_ub) { this->tight_upper_bound_ = tight_ub; }

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
                assum.add_lower_bound(zero());
                assum.add_upper_bound(one());
                assum.lower_bound_deprecated(zero());
                assum.upper_bound_deprecated(one());
                break;
            }
            case types::PrimitiveType::UInt8: {
                assum.add_lower_bound(integer(std::numeric_limits<uint8_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<uint8_t>::max()));
                assum.lower_bound_deprecated(integer(std::numeric_limits<uint8_t>::min()));
                assum.upper_bound_deprecated(integer(std::numeric_limits<uint8_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt16: {
                assum.add_lower_bound(integer(std::numeric_limits<uint16_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<uint16_t>::max()));
                assum.lower_bound_deprecated(integer(std::numeric_limits<uint16_t>::min()));
                assum.upper_bound_deprecated(integer(std::numeric_limits<uint16_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt32: {
                assum.add_lower_bound(integer(std::numeric_limits<uint32_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<uint32_t>::max()));
                assum.lower_bound_deprecated(integer(std::numeric_limits<uint32_t>::min()));
                assum.upper_bound_deprecated(integer(std::numeric_limits<uint32_t>::max()));
                break;
            }
            case types::PrimitiveType::UInt64: {
                assum.add_lower_bound(integer(std::numeric_limits<uint64_t>::min()));
                assum.add_upper_bound(SymEngine::Inf);
                assum.lower_bound_deprecated(integer(std::numeric_limits<uint64_t>::min()));
                assum.upper_bound_deprecated(SymEngine::Inf);
                break;
            }
            case types::PrimitiveType::UInt128: {
                assum.add_lower_bound(integer(0));
                assum.add_upper_bound(SymEngine::Inf);
                assum.lower_bound_deprecated(integer(0));
                assum.upper_bound_deprecated(SymEngine::Inf);
                break;
            }
            case types::PrimitiveType::Int8: {
                assum.add_lower_bound(integer(std::numeric_limits<int8_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<int8_t>::max()));
                assum.lower_bound_deprecated(integer(std::numeric_limits<int8_t>::min()));
                assum.upper_bound_deprecated(integer(std::numeric_limits<int8_t>::max()));
                break;
            }
            case types::PrimitiveType::Int16: {
                assum.add_lower_bound(integer(std::numeric_limits<int16_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<int16_t>::max()));
                assum.lower_bound_deprecated(integer(std::numeric_limits<int16_t>::min()));
                assum.upper_bound_deprecated(integer(std::numeric_limits<int16_t>::max()));
                break;
            }
            case types::PrimitiveType::Int32: {
                assum.add_lower_bound(integer(std::numeric_limits<int32_t>::min()));
                assum.add_upper_bound(integer(std::numeric_limits<int32_t>::max()));
                assum.lower_bound_deprecated(integer(std::numeric_limits<int32_t>::min()));
                assum.upper_bound_deprecated(integer(std::numeric_limits<int32_t>::max()));
                break;
            }
            case types::PrimitiveType::Int64: {
                assum.add_lower_bound(SymEngine::NegInf);
                assum.add_upper_bound(SymEngine::Inf);
                assum.lower_bound_deprecated(SymEngine::NegInf);
                assum.upper_bound_deprecated(SymEngine::Inf);
                break;
            }
            case types::PrimitiveType::Int128: {
                assum.add_lower_bound(SymEngine::NegInf);
                assum.add_upper_bound(SymEngine::Inf);
                assum.lower_bound_deprecated(SymEngine::NegInf);
                assum.upper_bound_deprecated(SymEngine::Inf);
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

} // namespace symbolic
} // namespace sdfg
