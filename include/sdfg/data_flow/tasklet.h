#pragma once

#include "sdfg/data_flow/code_node.h"
#include "sdfg/exceptions.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class FunctionBuilder;
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

enum TaskletCode {
    assign,
    // Arithmetic
    neg,
    add,
    sub,
    mul,
    div,
    fma,
    mod,
    max,
    min,
    minnum,
    maxnum,
    minimum,
    maximum,
    trunc,
    // Logical
    logical_and,
    logical_or,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    bitwise_not,
    shift_left,
    shift_right,
    olt,
    ole,
    oeq,
    one,
    oge,
    ogt,
    ord,
    ult,
    ule,
    ueq,
    une,
    uge,
    ugt,
    uno,
    // Math
    abs,
    acos,
    acosf,
    acosl,
    acosh,
    acoshf,
    acoshl,
    asin,
    asinf,
    asinl,
    asinh,
    asinhf,
    asinhl,
    atan,
    atanf,
    atanl,
    atan2,
    atan2f,
    atan2l,
    atanh,
    atanhf,
    atanhl,
    cabs,
    cabsf,
    cabsl,
    ceil,
    ceilf,
    ceill,
    copysign,
    copysignf,
    copysignl,
    cos,
    cosf,
    cosl,
    cosh,
    coshf,
    coshl,
    cbrt,
    cbrtf,
    cbrtl,
    erf,
    erff,
    erfl,
    exp10,
    exp10f,
    exp10l,
    exp2,
    exp2f,
    exp2l,
    exp,
    expf,
    expl,
    expm1,
    expm1f,
    expm1l,
    fabs,
    fabsf,
    fabsl,
    floor,
    floorf,
    floorl,
    fls,
    flsl,
    fmax,
    fmaxf,
    fmaxl,
    fmin,
    fminf,
    fminl,
    fmod,
    fmodf,
    fmodl,
    frexp,
    frexpf,
    frexpl,
    labs,
    ldexp,
    ldexpf,
    ldexpl,
    log10,
    log10f,
    log10l,
    log2,
    log2f,
    log2l,
    log,
    logf,
    logl,
    logb,
    logbf,
    logbl,
    log1p,
    log1pf,
    log1pl,
    modf,
    modff,
    modfl,
    nearbyint,
    nearbyintf,
    nearbyintl,
    pow,
    powf,
    powl,
    rint,
    rintf,
    rintl,
    lrint,
    llrint,
    round,
    roundf,
    roundl,
    lround,
    llround,
    roundeven,
    roundevenf,
    roundevenl,
    sin,
    sinf,
    sinl,
    sinh,
    sinhf,
    sinhl,
    sqrt,
    sqrtf,
    sqrtl,
    rsqrt,
    rsqrtf,
    rsqrtl,
    tan,
    tanf,
    tanl,
    tanh,
    tanhf,
    tanhl
};

constexpr size_t arity(TaskletCode c) {
    switch (c) {
        case TaskletCode::assign:
            return 1;
        // Arithmetic
        case TaskletCode::neg:
            return 1;
        case TaskletCode::add:
            return 2;
        case TaskletCode::sub:
            return 2;
        case TaskletCode::mul:
            return 2;
        case TaskletCode::div:
            return 2;
        case TaskletCode::fma:
            return 3;
        case TaskletCode::mod:
            return 2;
        case TaskletCode::max:
            return 2;
        case TaskletCode::min:
            return 2;
        case TaskletCode::minnum:
            return 2;
        case TaskletCode::maxnum:
            return 2;
        case TaskletCode::minimum:
            return 2;
        case TaskletCode::maximum:
            return 2;
        case TaskletCode::trunc:
            return 1;
        // Logical
        case TaskletCode::logical_and:
            return 2;
        case TaskletCode::logical_or:
            return 2;
        case TaskletCode::bitwise_and:
            return 2;
        case TaskletCode::bitwise_or:
            return 2;
        case TaskletCode::bitwise_xor:
            return 2;
        case TaskletCode::bitwise_not:
            return 1;
        case TaskletCode::shift_left:
            return 2;
        case TaskletCode::shift_right:
            return 2;
        case TaskletCode::olt:
            return 2;
        case TaskletCode::ole:
            return 2;
        case TaskletCode::oeq:
            return 2;
        case TaskletCode::one:
            return 2;
        case TaskletCode::oge:
            return 2;
        case TaskletCode::ogt:
            return 2;
        case TaskletCode::ord:
            return 2;
        case TaskletCode::ult:
            return 2;
        case TaskletCode::ule:
            return 2;
        case TaskletCode::ueq:
            return 2;
        case TaskletCode::une:
            return 2;
        case TaskletCode::uge:
            return 2;
        case TaskletCode::ugt:
            return 2;
        case TaskletCode::uno:
            return 2;
        // Math
        case TaskletCode::abs:
            return 1;
        case TaskletCode::acos:
            return 1;
        case TaskletCode::acosf:
            return 1;
        case TaskletCode::acosl:
            return 1;
        case TaskletCode::acosh:
            return 1;
        case TaskletCode::acoshf:
            return 1;
        case TaskletCode::acoshl:
            return 1;
        case TaskletCode::asin:
            return 1;
        case TaskletCode::asinf:
            return 1;
        case TaskletCode::asinl:
            return 1;
        case TaskletCode::asinh:
            return 1;
        case TaskletCode::asinhf:
            return 1;
        case TaskletCode::asinhl:
            return 1;
        case TaskletCode::atan:
            return 1;
        case TaskletCode::atanf:
            return 1;
        case TaskletCode::atanl:
            return 1;
        case TaskletCode::atan2:
            return 1;
        case TaskletCode::atan2f:
            return 1;
        case TaskletCode::atan2l:
            return 1;
        case TaskletCode::atanh:
            return 1;
        case TaskletCode::atanhf:
            return 1;
        case TaskletCode::atanhl:
            return 1;
        case TaskletCode::cabs:
            return 1;
        case TaskletCode::cabsf:
            return 1;
        case TaskletCode::cabsl:
            return 1;
        case TaskletCode::ceil:
            return 1;
        case TaskletCode::ceilf:
            return 1;
        case TaskletCode::ceill:
            return 1;
        case TaskletCode::copysign:
            return 1;
        case TaskletCode::copysignf:
            return 1;
        case TaskletCode::copysignl:
            return 1;
        case TaskletCode::cos:
            return 1;
        case TaskletCode::cosf:
            return 1;
        case TaskletCode::cosl:
            return 1;
        case TaskletCode::cosh:
            return 1;
        case TaskletCode::coshf:
            return 1;
        case TaskletCode::coshl:
            return 1;
        case TaskletCode::cbrt:
            return 1;
        case TaskletCode::cbrtf:
            return 1;
        case TaskletCode::cbrtl:
            return 1;
        case TaskletCode::erf:
            return 1;
        case TaskletCode::erff:
            return 1;
        case TaskletCode::erfl:
            return 1;
        case TaskletCode::exp10:
            return 1;
        case TaskletCode::exp10f:
            return 1;
        case TaskletCode::exp10l:
            return 1;
        case TaskletCode::exp2:
            return 1;
        case TaskletCode::exp2f:
            return 1;
        case TaskletCode::exp2l:
            return 1;
        case TaskletCode::exp:
            return 1;
        case TaskletCode::expf:
            return 1;
        case TaskletCode::expl:
            return 1;
        case TaskletCode::expm1:
            return 1;
        case TaskletCode::expm1f:
            return 1;
        case TaskletCode::expm1l:
            return 1;
        case TaskletCode::fabs:
            return 1;
        case TaskletCode::fabsf:
            return 1;
        case TaskletCode::fabsl:
            return 1;
        case TaskletCode::floor:
            return 1;
        case TaskletCode::floorf:
            return 1;
        case TaskletCode::floorl:
            return 1;
        case TaskletCode::fls:
            return 1;
        case TaskletCode::flsl:
            return 1;
        case TaskletCode::fmax:
            return 1;
        case TaskletCode::fmaxf:
            return 1;
        case TaskletCode::fmaxl:
            return 1;
        case TaskletCode::fmin:
            return 1;
        case TaskletCode::fminf:
            return 1;
        case TaskletCode::fminl:
            return 1;
        case TaskletCode::fmod:
            return 1;
        case TaskletCode::fmodf:
            return 1;
        case TaskletCode::fmodl:
            return 1;
        case TaskletCode::frexp:
            return 1;
        case TaskletCode::frexpf:
            return 1;
        case TaskletCode::frexpl:
            return 1;
        case TaskletCode::labs:
            return 1;
        case TaskletCode::ldexp:
            return 1;
        case TaskletCode::ldexpf:
            return 1;
        case TaskletCode::ldexpl:
            return 1;
        case TaskletCode::log10:
            return 1;
        case TaskletCode::log10f:
            return 1;
        case TaskletCode::log10l:
            return 1;
        case TaskletCode::log2:
            return 1;
        case TaskletCode::log2f:
            return 1;
        case TaskletCode::log2l:
            return 1;
        case TaskletCode::log:
            return 1;
        case TaskletCode::logf:
            return 1;
        case TaskletCode::logl:
            return 1;
        case TaskletCode::logb:
            return 1;
        case TaskletCode::logbf:
            return 1;
        case TaskletCode::logbl:
            return 1;
        case TaskletCode::log1p:
            return 1;
        case TaskletCode::log1pf:
            return 1;
        case TaskletCode::log1pl:
            return 1;
        case TaskletCode::modf:
            return 1;
        case TaskletCode::modff:
            return 1;
        case TaskletCode::modfl:
            return 1;
        case TaskletCode::nearbyint:
            return 1;
        case TaskletCode::nearbyintf:
            return 1;
        case TaskletCode::nearbyintl:
            return 1;
        case TaskletCode::pow:
            return 2;
        case TaskletCode::powf:
            return 2;
        case TaskletCode::powl:
            return 2;
        case TaskletCode::lrint:
            return 1;
        case TaskletCode::llrint:
            return 1;
        case TaskletCode::rint:
            return 1;
        case TaskletCode::rintf:
            return 1;
        case TaskletCode::rintl:
            return 1;
        case TaskletCode::round:
            return 1;
        case TaskletCode::roundf:
            return 1;
        case TaskletCode::roundl:
            return 1;
        case TaskletCode::lround:
            return 1;
        case TaskletCode::llround:
            return 1;
        case TaskletCode::roundeven:
            return 1;
        case TaskletCode::roundevenf:
            return 1;
        case TaskletCode::roundevenl:
            return 1;
        case TaskletCode::sin:
            return 1;
        case TaskletCode::sinf:
            return 1;
        case TaskletCode::sinl:
            return 1;
        case TaskletCode::sinh:
            return 1;
        case TaskletCode::sinhf:
            return 1;
        case TaskletCode::sinhl:
            return 1;
        case TaskletCode::sqrt:
            return 1;
        case TaskletCode::sqrtf:
            return 1;
        case TaskletCode::sqrtl:
            return 1;
        case TaskletCode::rsqrt:
            return 1;
        case TaskletCode::rsqrtf:
            return 1;
        case TaskletCode::rsqrtl:
            return 1;
        case TaskletCode::tan:
            return 1;
        case TaskletCode::tanf:
            return 1;
        case TaskletCode::tanl:
            return 1;
        case TaskletCode::tanh:
            return 1;
        case TaskletCode::tanhf:
            return 1;
        case TaskletCode::tanhl:
            return 1;
    };
    throw InvalidSDFGException("Invalid tasklet code");
};

constexpr bool is_infix(TaskletCode c) {
    switch (c) {
        case TaskletCode::add:
        case TaskletCode::sub:
        case TaskletCode::mul:
        case TaskletCode::div:
        case TaskletCode::mod:
        case TaskletCode::logical_and:
        case TaskletCode::logical_or:
        case TaskletCode::bitwise_and:
        case TaskletCode::bitwise_or:
        case TaskletCode::bitwise_xor:
        case TaskletCode::shift_left:
        case TaskletCode::shift_right:
        case TaskletCode::olt:
        case TaskletCode::ole:
        case TaskletCode::oeq:
        case TaskletCode::one:
        case TaskletCode::oge:
        case TaskletCode::ogt:
        case TaskletCode::ord:
        case TaskletCode::ult:
        case TaskletCode::ule:
        case TaskletCode::ueq:
        case TaskletCode::une:
        case TaskletCode::uge:
        case TaskletCode::ugt:
        case TaskletCode::uno:
            return true;
        default:
            return false;
    }
    throw InvalidSDFGException("Invalid tasklet code");
};

class Tasklet : public CodeNode {
    friend class sdfg::builder::FunctionBuilder;
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    TaskletCode code_;

    Tasklet(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const TaskletCode code,
        const std::string& output,
        const std::vector<std::string>& inputs
    );

public:
    Tasklet(const Tasklet& data_node) = delete;
    Tasklet& operator=(const Tasklet&) = delete;

    void validate(const Function& function) const override;

    TaskletCode code() const;

    const std::string& output() const;

    virtual std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};
} // namespace data_flow
} // namespace sdfg
