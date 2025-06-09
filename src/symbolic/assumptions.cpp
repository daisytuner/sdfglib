#include "sdfg/symbolic/assumptions.h"

#include "sdfg/types/scalar.h"

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
    return *this;
};

const Symbol& Assumption::symbol() const { return symbol_; };

const Expression& Assumption::lower_bound() const { return lower_bound_; };

void Assumption::lower_bound(const Expression& lower_bound) {
    lower_bound_ = lower_bound;
};

const Expression& Assumption::upper_bound() const { return upper_bound_; };

void Assumption::upper_bound(const Expression& upper_bound) {
    upper_bound_ = upper_bound;
};

Assumption Assumption::create(const symbolic::Symbol& symbol, const types::IType& type) {
    if (auto scalar_type = dynamic_cast<const types::Scalar*>(&type)) {
        auto assum = Assumption(symbol);

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
            case types::PrimitiveType::UInt128: {
                assum.lower_bound(symbolic::integer(0));
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
            case types::PrimitiveType::Int128: {
                assum.lower_bound(symbolic::infty(-1));
                assum.upper_bound(symbolic::infty(1));
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

}  // namespace symbolic
}  // namespace sdfg
