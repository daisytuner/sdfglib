#include "sdfg/codegen/language_extensions/c_language_extension.h"

#include "sdfg/codegen/utils.h"
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

std::string CLanguageExtension::primitive_type(const types::PrimitiveType prim_type) {
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
        case types::PrimitiveType::UInt8:
            return "char";
        case types::PrimitiveType::UInt16:
            return "unsigned short";
        case types::PrimitiveType::UInt32:
            return "unsigned int";
        case types::PrimitiveType::UInt64:
            return "unsigned long long";
        case types::PrimitiveType::Half:
            return "__fp16";
        case types::PrimitiveType::BFloat:
            return "__bf16";
        case types::PrimitiveType::Float:
            return "float";
        case types::PrimitiveType::Double:
            return "double";
        case types::PrimitiveType::X86_FP80:
            return "__float80";
        case types::PrimitiveType::FP128:
            return "__float128";
        case types::PrimitiveType::PPC_FP128:
            return "__float128";
    }

    throw std::runtime_error("Unknown primitive type");
};

std::string CLanguageExtension::declaration(const std::string& name, const types::IType& type,
                                            bool use_initializer) {
    std::stringstream val;

    if (auto scalar_type = dynamic_cast<const types::Scalar*>(&type)) {
        val << primitive_type(scalar_type->primitive_type());
        val << " ";
        val << name;
    } else if (auto array_type = dynamic_cast<const types::Array*>(&type)) {
        auto& element_type = array_type->element_type();
        val << declaration(name + "[" + this->expression(array_type->num_elements()) + "]",
                           element_type);
        if (array_type->alignment() > 1) {
            val << " __attribute__((aligned(" << array_type->alignment() << ")))";
        }
    } else if (auto pointer_type = dynamic_cast<const types::Pointer*>(&type)) {
        auto& pointee_type = pointer_type->pointee_type();
        val << declaration("(*" + name + ")", pointee_type);
    } else if (auto ref_type = dynamic_cast<const Reference*>(&type)) {
        val << declaration("&" + name, ref_type->reference_type());
    } else if (auto structure_type = dynamic_cast<const types::Structure*>(&type)) {
        val << structure_type->name();
        val << " ";
        val << name;
    } else if (auto function_type = dynamic_cast<const types::Function*>(&type)) {
        val << declaration("", function_type->return_type());
        val << " ";
        val << name;
        val << "(";
        for (size_t i = 0; i < function_type->num_params(); ++i) {
            val << declaration("", function_type->param_type(symbolic::integer(i)));
            if (i < function_type->num_params() - 1) {
                val << ", ";
            }
        }
        if (function_type->is_var_arg()) {
            if (function_type->num_params() > 0) {
                val << ", ";
            }
            val << "...";
        }
        val << ")";
    } else {
        throw std::runtime_error("Unknown declaration type");
    }

    if (use_initializer && !type.initializer().empty()) {
        val << " = " << type.initializer();
    }

    return val.str();
};

std::string CLanguageExtension::allocation(const std::string& name, const types::IType& type) {
    std::stringstream val;

    if (dynamic_cast<const types::Scalar*>(&type)) {
        val << declaration(name, type);
    } else if (auto array_type = dynamic_cast<const types::Array*>(&type)) {
        val << declaration(name + "[" + this->expression(array_type->num_elements()) + "]",
                           array_type->element_type());
    } else if (auto pointer_type = dynamic_cast<const types::Pointer*>(&type)) {
        val << declaration(name, type);
        val << " = (" << declaration("", type) << ") ";
        val << "malloc(";
        if (auto array_type = dynamic_cast<const types::Array*>(&pointer_type->pointee_type())) {
            val << this->expression(array_type->num_elements());
        } else {
            val << "1";
        }
        val << " * sizeof(";
        val << declaration("", pointer_type->pointee_type());
        val << "))";
    } else if (dynamic_cast<const types::Structure*>(&type)) {
        val << declaration(name, type);
    } else {
        throw std::runtime_error("Unknown allocation type");
    }

    return val.str();
};

std::string CLanguageExtension::deallocation(const std::string& name, const types::IType& type) {
    std::stringstream val;

    if (dynamic_cast<const types::Scalar*>(&type)) {
        // Do nothing
    } else if (dynamic_cast<const types::Array*>(&type)) {
        // Do nothing
    } else if (dynamic_cast<const types::Pointer*>(&type)) {
        val << "free(" << name << ")";
    } else if (dynamic_cast<const types::Structure*>(&type)) {
        // Do nothing
    } else {
        throw std::runtime_error("Unknown deallocation type");
    }

    return val.str();
};

std::string CLanguageExtension::type_cast(const std::string& name, const types::IType& type) {
    std::stringstream val;

    val << "(";
    val << declaration("", type);
    val << ") ";
    val << name;

    return val.str();
};

std::string CLanguageExtension::subset(const Function& function, const types::IType& type,
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

std::string CLanguageExtension::expression(const symbolic::Expression& expr) {
    CSymbolicPrinter printer;
    return printer.apply(expr);
};

std::string CLanguageExtension::tasklet(const data_flow::Tasklet& tasklet) {
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

std::string CLanguageExtension::library_node(const data_flow::LibraryNode& libnode) {
    data_flow::LibraryNodeType lib_node_type = libnode.call();
    switch (lib_node_type) {
        default:
            throw std::runtime_error("Unsupported library node type");
    }
}

void CSymbolicPrinter::bvisit(const SymEngine::Infty& x) {
    if (x.is_negative_infinity())
        str_ = "-INFINITY";
    else if (x.is_positive_infinity())
        str_ = "INFINITY";
};

void CSymbolicPrinter::bvisit(const SymEngine::BooleanAtom& x) {
    str_ = x.get_val() ? "true" : "false";
};

void CSymbolicPrinter::bvisit(const SymEngine::Symbol& x) {
    if (symbolic::is_nullptr(symbolic::symbol(x.get_name()))) {
        str_ = "NULL";
        return;
    } else if (symbolic::is_memory_address(symbolic::symbol(x.get_name()))) {
        // symbol of form reinterpret_cast<T*>(0x1234)
        str_ = x.get_name();
        return;
    }
    str_ = x.get_name();
};

void CSymbolicPrinter::bvisit(const SymEngine::And& x) {
    std::ostringstream s;
    auto container = x.get_container();
    s << apply(*container.begin());
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        s << " && " << apply(*it);
    }
    str_ = parenthesize(s.str());
};

void CSymbolicPrinter::bvisit(const SymEngine::Or& x) {
    std::ostringstream s;
    auto container = x.get_container();
    s << apply(*container.begin());
    for (auto it = ++(container.begin()); it != container.end(); ++it) {
        s << " || " << apply(*it);
    }
    str_ = parenthesize(s.str());
};

void CSymbolicPrinter::bvisit(const SymEngine::Not& x) {
    str_ = "!" + apply(x.get_arg());
    str_ = parenthesize(str_);
};

void CSymbolicPrinter::bvisit(const SymEngine::Equality& x) {
    str_ = apply(x.get_args()[0]) + " == " + apply(x.get_args()[1]);
    str_ = parenthesize(str_);
};

void CSymbolicPrinter::bvisit(const SymEngine::Unequality& x) {
    str_ = apply(x.get_args()[0]) + " != " + apply(x.get_args()[1]);
    str_ = parenthesize(str_);
};

void CSymbolicPrinter::bvisit(const SymEngine::Min& x) {
    std::ostringstream s;
    auto container = x.get_args();
    if (container.size() == 1) {
        s << apply(*container.begin());
    } else {
        s << "__daisy_min(";
        s << apply(*container.begin());

        // Recursively apply __daisy_min to the arguments
        SymEngine::vec_basic subargs;
        for (auto it = ++(container.begin()); it != container.end(); ++it) {
            subargs.push_back(*it);
        }
        auto submin = SymEngine::min(subargs);
        s << ", " << apply(submin);

        s << ")";
    }

    str_ = s.str();
};

void CSymbolicPrinter::bvisit(const SymEngine::Max& x) {
    std::ostringstream s;
    auto container = x.get_args();
    if (container.size() == 1) {
        s << apply(*container.begin());
    } else {
        s << "__daisy_max(";
        s << apply(*container.begin());

        // Recursively apply __daisy_max to the arguments
        SymEngine::vec_basic subargs;
        for (auto it = ++(container.begin()); it != container.end(); ++it) {
            subargs.push_back(*it);
        }
        auto submax = SymEngine::max(subargs);
        s << ", " << apply(submax);

        s << ")";
    }

    str_ = s.str();
};

void CSymbolicPrinter::_print_pow(std::ostringstream& o,
                                  const SymEngine::RCP<const SymEngine::Basic>& a,
                                  const SymEngine::RCP<const SymEngine::Basic>& b) {
    if (SymEngine::eq(*a, *SymEngine::E)) {
        o << "exp(" << apply(b) << ")";
    } else if (SymEngine::eq(*b, *SymEngine::rational(1, 2))) {
        o << "sqrt(" << apply(a) << ")";
    } else if (SymEngine::eq(*b, *SymEngine::rational(1, 3))) {
        o << "cbrt(" << apply(a) << ")";
    } else if (SymEngine::eq(*b, *SymEngine::integer(2))) {
        o << apply(a) + " * " + apply(a);
    } else {
        o << "pow(" << apply(a) << ", " << apply(b) << ")";
    }
};

}  // namespace codegen
}  // namespace sdfg
