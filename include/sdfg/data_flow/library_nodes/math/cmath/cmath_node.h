#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/types/type.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace cmath {

inline data_flow::LibraryNodeCode LibraryNodeType_CMath("CMath");
inline data_flow::LibraryNodeCode LibraryNodeType_CMath_Deprecated("Intrinsic");

/**
 * @enum CMathFunction
 * @brief Enumeration of C math library functions
 */
enum class CMathFunction {
    // Trigonometric functions
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    atan2,
    
    // Hyperbolic functions
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
    
    // Exponential and logarithmic functions
    exp,
    exp2,
    expm1,
    log,
    log10,
    log2,
    log1p,
    
    // Power functions
    pow,
    sqrt,
    cbrt,
    hypot,
    
    // Error and gamma functions
    erf,
    erfc,
    tgamma,
    lgamma,
    
    // Rounding and remainder functions
    ceil,
    floor,
    trunc,
    round,
    lround,
    llround,
    nearbyint,
    rint,
    lrint,
    llrint,
    fmod,
    remainder,
    
    // Floating-point manipulation functions
    frexp,
    ldexp,
    modf,
    scalbn,
    scalbln,
    ilogb,
    logb,
    nextafter,
    nexttoward,
    copysign,
    
    // Minimum, maximum, difference functions
    fmax,
    fmin,
    fdim,
    
    // Other functions
    fabs,
    abs,
    fma
};

/**
 * @brief Convert CMathFunction enum to function name stem (without type suffix)
 * @param func The CMathFunction enum value
 * @return The function name stem as a string
 */
constexpr const char* cmath_function_to_stem(CMathFunction func) {
    switch (func) {
        case CMathFunction::sin: return "sin";
        case CMathFunction::cos: return "cos";
        case CMathFunction::tan: return "tan";
        case CMathFunction::asin: return "asin";
        case CMathFunction::acos: return "acos";
        case CMathFunction::atan: return "atan";
        case CMathFunction::atan2: return "atan2";
        case CMathFunction::sinh: return "sinh";
        case CMathFunction::cosh: return "cosh";
        case CMathFunction::tanh: return "tanh";
        case CMathFunction::asinh: return "asinh";
        case CMathFunction::acosh: return "acosh";
        case CMathFunction::atanh: return "atanh";
        case CMathFunction::exp: return "exp";
        case CMathFunction::exp2: return "exp2";
        case CMathFunction::expm1: return "expm1";
        case CMathFunction::log: return "log";
        case CMathFunction::log10: return "log10";
        case CMathFunction::log2: return "log2";
        case CMathFunction::log1p: return "log1p";
        case CMathFunction::pow: return "pow";
        case CMathFunction::sqrt: return "sqrt";
        case CMathFunction::cbrt: return "cbrt";
        case CMathFunction::hypot: return "hypot";
        case CMathFunction::erf: return "erf";
        case CMathFunction::erfc: return "erfc";
        case CMathFunction::tgamma: return "tgamma";
        case CMathFunction::lgamma: return "lgamma";
        case CMathFunction::ceil: return "ceil";
        case CMathFunction::floor: return "floor";
        case CMathFunction::trunc: return "trunc";
        case CMathFunction::round: return "round";
        case CMathFunction::lround: return "lround";
        case CMathFunction::llround: return "llround";
        case CMathFunction::nearbyint: return "nearbyint";
        case CMathFunction::rint: return "rint";
        case CMathFunction::lrint: return "lrint";
        case CMathFunction::llrint: return "llrint";
        case CMathFunction::fmod: return "fmod";
        case CMathFunction::remainder: return "remainder";
        case CMathFunction::frexp: return "frexp";
        case CMathFunction::ldexp: return "ldexp";
        case CMathFunction::modf: return "modf";
        case CMathFunction::scalbn: return "scalbn";
        case CMathFunction::scalbln: return "scalbln";
        case CMathFunction::ilogb: return "ilogb";
        case CMathFunction::logb: return "logb";
        case CMathFunction::nextafter: return "nextafter";
        case CMathFunction::nexttoward: return "nexttoward";
        case CMathFunction::copysign: return "copysign";
        case CMathFunction::fmax: return "fmax";
        case CMathFunction::fmin: return "fmin";
        case CMathFunction::fdim: return "fdim";
        case CMathFunction::fabs: return "fabs";
        case CMathFunction::abs: return "abs";
        case CMathFunction::fma: return "fma";
    }
    return "";
}

/**
 * @brief Get the correct C math intrinsic name for a given function and primitive type
 *
 * Returns the appropriate intrinsic function name based on the primitive type:
 * - Float: adds 'f' suffix (e.g., "fmax" -> "fmaxf")
 * - Double: uses base name (e.g., "fmax" -> "fmax")
 * - Long Double/X86_FP80: adds 'l' suffix (e.g., "fmax" -> "fmaxl")
 *
 * @param func The CMathFunction enum value
 * @param prim_type The primitive type to generate the intrinsic for
 * @return The correct intrinsic function name
 */
inline std::string get_cmath_intrinsic_name(CMathFunction func, types::PrimitiveType prim_type) {
    std::string base_name = cmath_function_to_stem(func);
    
    switch (prim_type) {
        case types::PrimitiveType::Float:
            return base_name + "f";
        case types::PrimitiveType::Double:
            return base_name;
        case types::PrimitiveType::X86_FP80:
        case types::PrimitiveType::FP128:
        case types::PrimitiveType::PPC_FP128:
            return base_name + "l";
        case types::PrimitiveType::Half:
        case types::PrimitiveType::BFloat:
            // Half and BFloat are typically promoted to float for C math operations
            // as there are no standard Half/BFloat intrinsics in C math library
            return base_name + "f";
        default:
            // For integer types, return base name (though this shouldn't be called for integers)
            return base_name;
    }
}

class CMathNode : public math::MathNode {
private:
    CMathFunction function_;
    types::PrimitiveType primitive_type_;

public:
    CMathNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        CMathFunction function,
        types::PrimitiveType primitive_type,
        size_t arity
    );

    CMathFunction function() const;
    types::PrimitiveType primitive_type() const;
    std::string name() const;

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override {
        return false;
    };

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    virtual symbolic::Expression flop() const override;
};

class CMathNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class CMathNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    CMathNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const CMathNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace cmath
} // namespace math
} // namespace sdfg
