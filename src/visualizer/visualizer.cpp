#include "sdfg/visualizer/visualizer.h"

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/schedule.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/kernel.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"
#include "symengine/basic.h"

namespace sdfg {
namespace visualizer {

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
            return "fma";
        case data_flow::TaskletCode::mod:
            return "%";
        case data_flow::TaskletCode::max:
            return "max";
        case data_flow::TaskletCode::min:
            return "min";
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
    throw InvalidSDFGException("code_to_string: Unsupported tasklet code");
};

std::string Visualizer::expression(const std::string expr) {
    if (this->replacements_.empty()) return expr;
    std::string res = expr;
    size_t pos1 = 0, pos2 = 0;
    for (std::pair<const std::string, const std::string> replace : this->replacements_) {
        pos2 = res.find(replace.first);
        if (pos2 == res.npos) continue;
        pos1 = 0;
        std::stringstream res_tmp;
        while (pos2 < res.npos) {
            res_tmp << res.substr(pos1, pos2 - pos1) << replace.second;
            pos1 = pos2 + replace.first.size();
            pos2 = res.find(replace.first, pos1);
        }
        if (pos1 < res.npos) res_tmp << res.substr(pos1);
        res = res_tmp.str();
    }
    return res;
}

void Visualizer::visualizeNode(Schedule& schedule, structured_control_flow::ControlFlowNode& node) {
    if (auto block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        this->visualizeBlock(schedule, *block);
        return;
    }
    if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        this->visualizeSequence(schedule, *sequence);
        return;
    }
    if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        this->visualizeIfElse(schedule, *if_else);
        return;
    }
    if (auto while_loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        this->visualizeWhile(schedule, *while_loop);
        return;
    }
    if (auto loop = dynamic_cast<structured_control_flow::For*>(&node)) {
        this->visualizeFor(schedule, *loop);
        return;
    }
    if (auto return_node = dynamic_cast<structured_control_flow::Return*>(&node)) {
        this->visualizeReturn(schedule, *return_node);
        return;
    }
    if (auto break_node = dynamic_cast<structured_control_flow::Break*>(&node)) {
        this->visualizeBreak(schedule, *break_node);
        return;
    }
    if (auto continue_node = dynamic_cast<structured_control_flow::Continue*>(&node)) {
        this->visualizeContinue(schedule, *continue_node);
        return;
    }
    if (auto kernel_node = dynamic_cast<structured_control_flow::Kernel*>(&node)) {
        this->visualizeKernel(schedule, *kernel_node);
        return;
    }
    throw std::runtime_error("Unsupported control flow node");
}

void Visualizer::visualizeTasklet(data_flow::Tasklet const& tasklet) {
    std::string op = code_to_string(tasklet.code());
    std::vector<std::string> arguments;
    for (size_t i = 0; i < tasklet.inputs().size(); ++i) {
        std::string arg = tasklet.input(i).first;
        if (!tasklet.needs_connector(i)) {
            if (arg != "NAN" && arg != "INFINITY") {
                if (tasklet.input(i).second.primitive_type() == types::PrimitiveType::Float)
                    arg += "f";
            }
        }
        arguments.push_back(this->expression(arg));
    }

    if (tasklet.code() == data_flow::TaskletCode::assign) {
        this->stream_ << arguments.at(0);
    } else if (tasklet.code() == data_flow::TaskletCode::fma) {
        if (arguments.size() != 3) throw std::runtime_error("FMA requires 3 arguments");
        this->stream_ << arguments.at(0) << " * " << arguments.at(1) << " + " << arguments.at(2);
    } else if (data_flow::is_infix(tasklet.code())) {
        switch (data_flow::arity(tasklet.code())) {
            case 1:
                this->stream_ << op << " " << arguments.at(0);
                break;
            case 2:
                this->stream_ << arguments.at(0) << " " << op << " " << arguments.at(1);
                break;
            default:
                throw std::runtime_error("Unsupported arity");
        }
    } else {
        this->stream_ << op << "(" << helpers::join(arguments, ", ") << ")";
    }
}

void Visualizer::visualizeForBounds(symbolic::Symbol const& indvar,
                                    symbolic::Expression const& init,
                                    symbolic::Condition const& condition,
                                    symbolic::Expression const& update) {
    if ((init->get_type_code() == SymEngine::TypeID::SYMENGINE_INTEGER ||
         init->get_type_code() == SymEngine::TypeID::SYMENGINE_SYMBOL) &&
        (condition->get_type_code() == SymEngine::TypeID::SYMENGINE_STRICTLESSTHAN ||
         condition->get_type_code() == SymEngine::TypeID::SYMENGINE_LESSTHAN) &&
        condition->get_args().size() == 2 &&
        condition->get_args().at(0)->__str__() == indvar->__str__() &&
        (condition->get_args().at(1)->get_type_code() == SymEngine::TypeID::SYMENGINE_INTEGER ||
         condition->get_args().at(1)->get_type_code() == SymEngine::TypeID::SYMENGINE_SYMBOL) &&
        update->get_type_code() == SymEngine::TypeID::SYMENGINE_ADD &&
        update->get_args().size() == 2 &&
        (update->get_args().at(0)->__str__() == indvar->__str__() ||
         update->get_args().at(1)->__str__() == indvar->__str__()) &&
        (update->get_args().at(0)->get_type_code() == SymEngine::TypeID::SYMENGINE_INTEGER ||
         update->get_args().at(0)->get_type_code() == SymEngine::TypeID::SYMENGINE_SYMBOL ||
         update->get_args().at(1)->get_type_code() == SymEngine::TypeID::SYMENGINE_INTEGER ||
         update->get_args().at(1)->get_type_code() == SymEngine::TypeID::SYMENGINE_SYMBOL)) {
        this->stream_ << indvar->get_name() << " = " << init->__str__() << ":";
        if (condition->get_type_code() == SymEngine::TypeID::SYMENGINE_STRICTLESSTHAN)
            this->stream_ << "(" << condition->get_args().at(1)->__str__() << "-1)";
        else
            this->stream_ << condition->get_args().at(1)->__str__();
        size_t i = (update->get_args().at(0).get() == indvar.get()) ? 1 : 0;
        if (update->get_args().at(i)->__str__() != "1")
            this->stream_ << ":" << update->get_args().at(i)->__str__();
    } else {
        this->stream_ << indvar->get_name() << " = " << this->expression(init->__str__()) << "; "
                      << this->expression(condition->__str__()) << "; " << indvar->get_name()
                      << " = " << this->expression(update->__str__());
    }
}

void Visualizer::visualizeLibraryNode(const data_flow::LibraryNodeType libnode_type) {
    switch (libnode_type) {
        case data_flow::LibraryNodeType::LocalBarrier:
            this->stream_ << "Local Barrier";
            break;
        default:
            throw std::runtime_error("Unsupported library node type");
    }
}

void Visualizer::visualizeSubset(Function const& function, types::IType const& type,
                                 data_flow::Subset const& sub) {
    if (sub.empty()) return;
    if (dynamic_cast<const types::Scalar*>(&type)) {
        return;
    } else if (auto array_type = dynamic_cast<const types::Array*>(&type)) {
        this->stream_ << "[" << this->expression(sub.at(0)->__str__()) << "]";
        if (sub.size() > 1) {
            data_flow::Subset element_subset(sub.begin() + 1, sub.end());
            types::IType const& element_type = array_type->element_type();
            this->visualizeSubset(function, element_type, element_subset);
        }
    } else if (auto pointer_type = dynamic_cast<const types::Pointer*>(&type)) {
        this->stream_ << "[" << this->expression(sub.at(0)->__str__()) << "]";
        data_flow::Subset element_subset(sub.begin() + 1, sub.end());
        types::IType const& pointee_type = pointer_type->pointee_type();
        this->visualizeSubset(function, pointee_type, element_subset);
    } else if (auto structure_type = dynamic_cast<const types::Structure*>(&type)) {
        types::StructureDefinition const& definition = function.structure(structure_type->name());
        this->stream_ << ".member_" << this->expression(sub.at(0)->__str__());
        if (sub.size() > 1) {
            auto member = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(sub.at(0));
            types::IType const& member_type = definition.member_type(member);
            data_flow::Subset element_subset(sub.begin() + 1, sub.end());
            this->visualizeSubset(function, member_type, element_subset);
        }
    } else {
        throw InvalidSDFGException("visualizeSubset: Unsupported type");
    }
}

}  // namespace visualizer
}  // namespace sdfg
