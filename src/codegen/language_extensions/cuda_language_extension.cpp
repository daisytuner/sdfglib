#include "sdfg/codegen/language_extensions/cuda_language_extension.h"

#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/tasklet.h"

namespace sdfg {
namespace codegen {

constexpr const char* code_to_string(data_flow::TaskletCode c) {
    switch (c) {
        case data_flow::TaskletCode::assign:
            return "=";
        case data_flow::TaskletCode::neg:
            return "-";
        case data_flow::TaskletCode::add:
            return "+";
        case data_flow::TaskletCode::sub:
            return "-";
        case data_flow::TaskletCode::mul:
            return "*";
        case data_flow::TaskletCode::div:
            return "/";
        case data_flow::TaskletCode::fma:
            return "__daisy_fma";
        case data_flow::TaskletCode::mod:
            return "%";
        case data_flow::TaskletCode::max:
            return "__daisy_max";
        case data_flow::TaskletCode::min:
            return "__daisy_min";
        case data_flow::TaskletCode::minnum:
            return "minnum";
        case data_flow::TaskletCode::maxnum:
            return "maxnum";
        case data_flow::TaskletCode::minimum:
            return "minimum";
        case data_flow::TaskletCode::maximum:
            return "maximum";
        case data_flow::TaskletCode::trunc:
            return "trunc";
        case data_flow::TaskletCode::logical_and:
            return "&&";
        case data_flow::TaskletCode::logical_or:
            return "||";
        case data_flow::TaskletCode::bitwise_and:
            return "&";
        case data_flow::TaskletCode::bitwise_or:
            return "|";
        case data_flow::TaskletCode::bitwise_xor:
            return "^";
        case data_flow::TaskletCode::bitwise_not:
            return "~";
        case data_flow::TaskletCode::shift_left:
            return "<<";
        case data_flow::TaskletCode::shift_right:
            return ">>";
        case data_flow::TaskletCode::olt:
            return "<";
        case data_flow::TaskletCode::ole:
            return "<=";
        case data_flow::TaskletCode::oeq:
            return "==";
        case data_flow::TaskletCode::one:
            return "!=";
        case data_flow::TaskletCode::oge:
            return ">=";
        case data_flow::TaskletCode::ogt:
            return ">";
        case data_flow::TaskletCode::ord:
            return "==";
        case data_flow::TaskletCode::ult:
            return "<";
        case data_flow::TaskletCode::ule:
            return "<=";
        case data_flow::TaskletCode::ueq:
            return "==";
        case data_flow::TaskletCode::une:
            return "!=";
        case data_flow::TaskletCode::uge:
            return ">=";
        case data_flow::TaskletCode::ugt:
            return ">";
        case data_flow::TaskletCode::uno:
            return "!=";
        case data_flow::TaskletCode::abs:
            return "abs";
        case data_flow::TaskletCode::acos:
            return "acos";
        case data_flow::TaskletCode::acosf:
            return "acosf";
        case data_flow::TaskletCode::acosl:
            return "acosl";
        case data_flow::TaskletCode::acosh:
            return "acosh";
        case data_flow::TaskletCode::acoshf:
            return "acoshf";
        case data_flow::TaskletCode::acoshl:
            return "acoshl";
        case data_flow::TaskletCode::asin:
            return "asin";
        case data_flow::TaskletCode::asinf:
            return "asinf";
        case data_flow::TaskletCode::asinl:
            return "asinl";
        case data_flow::TaskletCode::asinh:
            return "asinh";
        case data_flow::TaskletCode::asinhf:
            return "asinhf";
        case data_flow::TaskletCode::asinhl:
            return "asinhl";
        case data_flow::TaskletCode::atan:
            return "atan";
        case data_flow::TaskletCode::atanf:
            return "atanf";
        case data_flow::TaskletCode::atanl:
            return "atanl";
        case data_flow::TaskletCode::atan2:
            return "atan2";
        case data_flow::TaskletCode::atan2f:
            return "atan2f";
        case data_flow::TaskletCode::atan2l:
            return "atan2l";
        case data_flow::TaskletCode::atanh:
            return "atanh";
        case data_flow::TaskletCode::atanhf:
            return "atanhf";
        case data_flow::TaskletCode::atanhl:
            return "atanhl";
        case data_flow::TaskletCode::cabs:
            return "cabs";
        case data_flow::TaskletCode::cabsf:
            return "cabsf";
        case data_flow::TaskletCode::cabsl:
            return "cabsl";
        case data_flow::TaskletCode::ceil:
            return "ceil";
        case data_flow::TaskletCode::ceilf:
            return "ceilf";
        case data_flow::TaskletCode::ceill:
            return "ceill";
        case data_flow::TaskletCode::copysign:
            return "copysign";
        case data_flow::TaskletCode::copysignf:
            return "copysignf";
        case data_flow::TaskletCode::copysignl:
            return "copysignl";
        case data_flow::TaskletCode::cos:
            return "cos";
        case data_flow::TaskletCode::cosf:
            return "cosf";
        case data_flow::TaskletCode::cosl:
            return "cosl";
        case data_flow::TaskletCode::cosh:
            return "cosh";
        case data_flow::TaskletCode::coshf:
            return "coshf";
        case data_flow::TaskletCode::coshl:
            return "coshl";
        case data_flow::TaskletCode::cbrt:
            return "cbrt";
        case data_flow::TaskletCode::cbrtf:
            return "cbrtf";
        case data_flow::TaskletCode::cbrtl:
            return "cbrtl";
        case data_flow::TaskletCode::exp10:
            return "exp10";
        case data_flow::TaskletCode::exp10f:
            return "exp10f";
        case data_flow::TaskletCode::exp10l:
            return "exp10l";
        case data_flow::TaskletCode::exp2:
            return "exp2";
        case data_flow::TaskletCode::exp2f:
            return "exp2f";
        case data_flow::TaskletCode::exp2l:
            return "exp2l";
        case data_flow::TaskletCode::exp:
            return "exp";
        case data_flow::TaskletCode::expf:
            return "expf";
        case data_flow::TaskletCode::expl:
            return "expl";
        case data_flow::TaskletCode::expm1:
            return "expm1";
        case data_flow::TaskletCode::expm1f:
            return "expm1f";
        case data_flow::TaskletCode::expm1l:
            return "expm1l";
        case data_flow::TaskletCode::fabs:
            return "fabs";
        case data_flow::TaskletCode::fabsf:
            return "fabsf";
        case data_flow::TaskletCode::fabsl:
            return "fabsl";
        case data_flow::TaskletCode::floor:
            return "floor";
        case data_flow::TaskletCode::floorf:
            return "floorf";
        case data_flow::TaskletCode::floorl:
            return "floorl";
        case data_flow::TaskletCode::fls:
            return "fls";
        case data_flow::TaskletCode::flsl:
            return "flsl";
        case data_flow::TaskletCode::fmax:
            return "fmax";
        case data_flow::TaskletCode::fmaxf:
            return "fmaxf";
        case data_flow::TaskletCode::fmaxl:
            return "fmaxl";
        case data_flow::TaskletCode::fmin:
            return "fmin";
        case data_flow::TaskletCode::fminf:
            return "fminf";
        case data_flow::TaskletCode::fminl:
            return "fminl";
        case data_flow::TaskletCode::fmod:
            return "fmod";
        case data_flow::TaskletCode::fmodf:
            return "fmodf";
        case data_flow::TaskletCode::fmodl:
            return "fmodl";
        case data_flow::TaskletCode::frexp:
            return "frexp";
        case data_flow::TaskletCode::frexpf:
            return "frexpf";
        case data_flow::TaskletCode::frexpl:
            return "frexpl";
        case data_flow::TaskletCode::labs:
            return "labs";
        case data_flow::TaskletCode::ldexp:
            return "ldexp";
        case data_flow::TaskletCode::ldexpf:
            return "ldexpf";
        case data_flow::TaskletCode::ldexpl:
            return "ldexpl";
        case data_flow::TaskletCode::log10:
            return "log10";
        case data_flow::TaskletCode::log10f:
            return "log10f";
        case data_flow::TaskletCode::log10l:
            return "log10l";
        case data_flow::TaskletCode::log2:
            return "log2";
        case data_flow::TaskletCode::log2f:
            return "log2f";
        case data_flow::TaskletCode::log2l:
            return "log2l";
        case data_flow::TaskletCode::log:
            return "log";
        case data_flow::TaskletCode::logf:
            return "logf";
        case data_flow::TaskletCode::logl:
            return "logl";
        case data_flow::TaskletCode::logb:
            return "logb";
        case data_flow::TaskletCode::logbf:
            return "logbf";
        case data_flow::TaskletCode::logbl:
            return "logbl";
        case data_flow::TaskletCode::log1p:
            return "log1p";
        case data_flow::TaskletCode::log1pf:
            return "log1pf";
        case data_flow::TaskletCode::log1pl:
            return "log1pl";
        case data_flow::TaskletCode::modf:
            return "modf";
        case data_flow::TaskletCode::modff:
            return "modff";
        case data_flow::TaskletCode::modfl:
            return "modfl";
        case data_flow::TaskletCode::nearbyint:
            return "nearbyint";
        case data_flow::TaskletCode::nearbyintf:
            return "nearbyintf";
        case data_flow::TaskletCode::nearbyintl:
            return "nearbyintl";
        case data_flow::TaskletCode::pow:
            return "pow";
        case data_flow::TaskletCode::powf:
            return "powf";
        case data_flow::TaskletCode::powl:
            return "powl";
        case data_flow::TaskletCode::rint:
            return "rint";
        case data_flow::TaskletCode::rintf:
            return "rintf";
        case data_flow::TaskletCode::rintl:
            return "rintl";
        case data_flow::TaskletCode::round:
            return "round";
        case data_flow::TaskletCode::roundf:
            return "roundf";
        case data_flow::TaskletCode::roundl:
            return "roundl";
        case data_flow::TaskletCode::roundeven:
            return "roundeven";
        case data_flow::TaskletCode::roundevenf:
            return "roundevenf";
        case data_flow::TaskletCode::roundevenl:
            return "roundevenl";
        case data_flow::TaskletCode::sin:
            return "sin";
        case data_flow::TaskletCode::sinf:
            return "sinf";
        case data_flow::TaskletCode::sinl:
            return "sinl";
        case data_flow::TaskletCode::sinh:
            return "sinh";
        case data_flow::TaskletCode::sinhf:
            return "sinhf";
        case data_flow::TaskletCode::sinhl:
            return "sinhl";
        case data_flow::TaskletCode::sqrt:
            return "sqrt";
        case data_flow::TaskletCode::sqrtf:
            return "sqrtf";
        case data_flow::TaskletCode::sqrtl:
            return "sqrtl";
        case data_flow::TaskletCode::rsqrt:
            return "rsqrt";
        case data_flow::TaskletCode::rsqrtf:
            return "rsqrtf";
        case data_flow::TaskletCode::rsqrtl:
            return "rsqrtl";
        case data_flow::TaskletCode::tan:
            return "tan";
        case data_flow::TaskletCode::tanf:
            return "tanf";
        case data_flow::TaskletCode::tanl:
            return "tanl";
        case data_flow::TaskletCode::tanh:
            return "tanh";
        case data_flow::TaskletCode::tanhf:
            return "tanhf";
        case data_flow::TaskletCode::tanhl:
            return "tanhl";
    };
    throw std::invalid_argument("Invalid tasklet code");
};

std::string CUDALanguageExtension::primitive_type(const types::PrimitiveType prim_type) {
    switch (prim_type) {
        case types::PrimitiveType::Void:
            return "void";
        case types::PrimitiveType::Bool:
            return "bool";
        case types::PrimitiveType::Int8:
            return "signed char";
        case types::PrimitiveType::Int16:
            return "short";
        case types::PrimitiveType::Int32:
            return "int";
        case types::PrimitiveType::Int64:
            return "long long";
        case types::PrimitiveType::Int128:
            return "__int128";
        case types::PrimitiveType::UInt8:
            return "char";
        case types::PrimitiveType::UInt16:
            return "unsigned short";
        case types::PrimitiveType::UInt32:
            return "unsigned int";
        case types::PrimitiveType::UInt64:
            return "unsigned long long";
        case types::PrimitiveType::UInt128:
            return "unsigned __int128";
        case types::PrimitiveType::Half:
            return "__fp16";
        case types::PrimitiveType::BFloat:
            return "__bf16";
        case types::PrimitiveType::Float:
            return "float";
        case types::PrimitiveType::Double:
            return "double";
        case types::PrimitiveType::X86_FP80:
            return "long double";
        case types::PrimitiveType::FP128:
            return "__float128";
        case types::PrimitiveType::PPC_FP128:
            return "__float128";
    }

    throw std::runtime_error("Unknown primitive type");
};

std::string CUDALanguageExtension::declaration(const std::string& name, const types::IType& type,
                                               bool use_initializer, bool use_alignment) {
    std::stringstream val;

    if (auto scalar_type = dynamic_cast<const types::Scalar*>(&type)) {
        if (scalar_type->storage_type() == types::StorageType_NV_Shared) {
            val << "__shared__ ";
        } else if (scalar_type->storage_type() == types::StorageType_NV_Constant) {
            val << "__constant__ ";
        }
        val << primitive_type(scalar_type->primitive_type());
        val << " ";
        val << name;
    } else if (auto array_type = dynamic_cast<const types::Array*>(&type)) {
        auto& element_type = array_type->element_type();
        val << declaration(name + "[" + this->expression(array_type->num_elements()) + "]",
                           element_type);
    } else if (auto pointer_type = dynamic_cast<const types::Pointer*>(&type)) {
        const types::IType& pointee = pointer_type->pointee_type();

        const bool pointee_is_function_or_array = dynamic_cast<const types::Function*>(&pointee) ||
                                                  dynamic_cast<const types::Array*>(&pointee);

        // Parenthesise *only* when it is needed to bind tighter than [] or ()
        std::string decorated = pointee_is_function_or_array ? "(*" + name + ")" : "*" + name;

        val << declaration(decorated, pointee);
    } else if (auto ref_type = dynamic_cast<const Reference*>(&type)) {
        val << declaration("&" + name, ref_type->reference_type());
    } else if (auto structure_type = dynamic_cast<const types::Structure*>(&type)) {
        if (structure_type->storage_type() == types::StorageType_NV_Shared) {
            val << "__shared__ ";
        } else if (structure_type->storage_type() == types::StorageType_NV_Constant) {
            val << "__constant__ ";
        }
        val << structure_type->name();
        val << " ";
        val << name;
    } else if (auto function_type = dynamic_cast<const types::Function*>(&type)) {
        std::stringstream params;
        for (size_t i = 0; i < function_type->num_params(); ++i) {
            params << declaration("", function_type->param_type(symbolic::integer(i)));
            if (i + 1 < function_type->num_params()) params << ", ";
        }
        if (function_type->is_var_arg()) {
            if (function_type->num_params() > 0) params << ", ";
            params << "...";
        }

        const std::string fun_name = name + "(" + params.str() + ")";
        val << declaration(fun_name, function_type->return_type());
    } else {
        throw std::runtime_error("Unknown declaration type");
    }

    if (use_alignment && type.alignment() > 0) {
        val << " __attribute__((aligned(" << type.alignment() << ")))";
    }

    if (use_initializer && !type.initializer().empty()) {
        val << " = " << type.initializer();
    }

    return val.str();
};

std::string CUDALanguageExtension::type_cast(const std::string& name, const types::IType& type) {
    std::stringstream val;

    val << "reinterpret_cast";
    val << "<";
    val << declaration("", type);
    val << ">";
    val << "(" << name << ")";

    return val.str();
};

std::string CUDALanguageExtension::subset(const Function& function, const types::IType& type,
                                          const data_flow::Subset& sub) {
    if (sub.empty()) {
        return "";
    }

    if (dynamic_cast<const types::Scalar*>(&type)) {
        return "";
    } else if (auto array_type = dynamic_cast<const types::Array*>(&type)) {
        std::string subset_str = "[" + this->expression(sub.at(0)) + "]";

        if (sub.size() > 1) {
            data_flow::Subset element_subset(sub.begin() + 1, sub.end());
            auto& element_type = array_type->element_type();
            return subset_str + subset(function, element_type, element_subset);
        } else {
            return subset_str;
        }
    } else if (auto pointer_type = dynamic_cast<const types::Pointer*>(&type)) {
        std::string subset_str = "[" + this->expression(sub.at(0)) + "]";

        data_flow::Subset element_subset(sub.begin() + 1, sub.end());
        auto& pointee_type = pointer_type->pointee_type();
        return subset_str + subset(function, pointee_type, element_subset);
    } else if (auto structure_type = dynamic_cast<const types::Structure*>(&type)) {
        auto& definition = function.structure(structure_type->name());

        std::string subset_str = ".member_" + this->expression(sub.at(0));
        if (sub.size() > 1) {
            auto member = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(sub.at(0));
            auto& member_type = definition.member_type(member);
            data_flow::Subset element_subset(sub.begin() + 1, sub.end());
            return subset_str + subset(function, member_type, element_subset);
        } else {
            return subset_str;
        }
    }

    throw std::invalid_argument("Invalid subset type");
};

std::string CUDALanguageExtension::expression(const symbolic::Expression& expr) {
    CPPSymbolicPrinter printer;
    return printer.apply(expr);
};

std::string CUDALanguageExtension::tasklet(const data_flow::Tasklet& tasklet) {
    std::string op = code_to_string(tasklet.code());
    std::vector<std::string> arguments;
    for (size_t i = 0; i < tasklet.inputs().size(); ++i) {
        std::string arg = tasklet.input(i).first;
        if (!tasklet.needs_connector(i)) {
            if (arg != "NAN" && arg != "INFINITY") {
                if (tasklet.input(i).second.primitive_type() == types::PrimitiveType::Float) {
                    arg += "f";
                }
            }
        }
        arguments.push_back(arg);
    }

    if (tasklet.code() == data_flow::TaskletCode::assign) {
        return arguments.at(0);
    } else if (data_flow::is_infix(tasklet.code())) {
        switch (data_flow::arity(tasklet.code())) {
            case 1:
                return op + arguments.at(0);
            case 2:
                return arguments.at(0) + " " + op + " " + arguments.at(1);
            default:
                throw std::runtime_error("Unsupported arity");
        }
    } else {
        return op + "(" + helpers::join(arguments, ", ") + ")";
    }
};

std::string CUDALanguageExtension::zero(const types::PrimitiveType prim_type) {
    switch (prim_type) {
        case types::Void:
            throw InvalidSDFGException("No zero for void type possible");
        case types::Bool:
            return "false";
        case types::Int8:
            return "0";
        case types::Int16:
            return "0";
        case types::Int32:
            return "0";
        case types::Int64:
            return "0ll";
        case types::Int128:
            return "0";
        case types::UInt8:
            return "0u";
        case types::UInt16:
            return "0u";
        case types::UInt32:
            return "0u";
        case types::UInt64:
            return "0ull";
        case types::UInt128:
            return "0";
        case types::Half:
            return "CUDART_ZERO_FP16";
        case types::BFloat:
            return "CUDART_ZERO_BF16";
        case types::Float:
            return "0.0f";
        case types::Double:
            return "0.0";
        case types::X86_FP80:
            return "0.0l";
        case types::FP128:
            return "0.0";
        case types::PPC_FP128:
            return "0.0";
    }
}

}  // namespace codegen
}  // namespace sdfg
