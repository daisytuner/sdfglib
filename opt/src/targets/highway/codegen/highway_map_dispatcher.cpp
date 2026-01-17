#include "sdfg/targets/highway/codegen/highway_map_dispatcher.h"

#include "sdfg/targets/highway/schedule.h"
#include "sdfg/transformations/highway_transform.h"

#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/analysis/users.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <sdfg/codegen/instrumentation/instrumentation_info.h>
#include <sdfg/helpers/helpers.h>
#include <sdfg/structured_control_flow/map.h>

namespace sdfg {
namespace highway {

HighwayMapDispatcher::HighwayMapDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node), indvar_(node.indvar()), arguments_(), arguments_declaration_(), arguments_lookup_(), locals_(),
      local_symbols_(), vec_type_(types::PrimitiveType::Float) {
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();
    for (auto& entry : arguments_analysis.arguments(analysis_manager, node_)) {
        arguments_.push_back(entry.first);
        arguments_lookup_.insert(entry.first);

        auto& type = sdfg_.type(entry.first);
        arguments_declaration_.push_back(this->language_extension_.declaration(entry.first, type));
    }
    for (auto& local : arguments_analysis.locals(analysis_manager, node_)) {
        locals_.insert(local);
        if (local == indvar_->get_name()) {
            continue;
        }

        auto& type = sdfg_.type(local);
        if (types::is_integer(type.primitive_type())) {
            local_symbols_.insert(symbolic::symbol(local));
        }
    }
};

void HighwayMapDispatcher::dispatch_kernel_call(codegen::PrettyPrinter& main_stream, const std::string& kernel_name) {
    main_stream << kernel_name << "(" << helpers::join(arguments_, ", ") << ");" << std::endl;
};

void HighwayMapDispatcher::
    dispatch_kernel_declaration(codegen::PrettyPrinter& globals_stream, const std::string& kernel_name) {
    if (this->language_extension_.language() == "C") {
        globals_stream << "void " << kernel_name << "(" << helpers::join(arguments_declaration_, ", ") << ");"
                       << std::endl;
    } else {
        globals_stream << "extern \"C\" void " << kernel_name << "(" << helpers::join(arguments_declaration_, ", ")
                       << ");" << std::endl;
    }
};

void HighwayMapDispatcher::dispatch_kernel_preamble(codegen::PrettyPrinter& library_stream, const std::string& kernel_name) {
    library_stream << "HWY_ATTR void ";
    library_stream << kernel_name;
    library_stream << "(";
    library_stream << helpers::join(arguments_declaration_, ", ");
    library_stream << ")" << std::endl;
};

void HighwayMapDispatcher::
    dispatch_kernel_body(codegen::CodeSnippetFactory& library_snippet_factory, codegen::PrettyPrinter& library_stream) {
    library_stream << "const hn::ScalableTag<uint8_t> daisy_vec_u8;" << std::endl;
    library_stream << "const hn::ScalableTag<uint16_t> daisy_vec_u16;" << std::endl;
    library_stream << "const hn::ScalableTag<uint32_t> daisy_vec_u32;" << std::endl;
    library_stream << "const hn::ScalableTag<uint64_t> daisy_vec_u64;" << std::endl;
    library_stream << "const hn::ScalableTag<int8_t> daisy_vec_s8;" << std::endl;
    library_stream << "const hn::ScalableTag<int16_t> daisy_vec_s16;" << std::endl;
    library_stream << "const hn::ScalableTag<int32_t> daisy_vec_s32;" << std::endl;
    library_stream << "const hn::ScalableTag<int64_t> daisy_vec_s64;" << std::endl;
    library_stream << "const hn::ScalableTag<float> daisy_vec_f32;" << std::endl;
    library_stream << "const hn::ScalableTag<double> daisy_vec_f64;" << std::endl;
    library_stream << std::endl;

    auto& indvar_type = sdfg_.type(indvar_->get_name());
    library_stream << this->language_extension_.declaration(indvar_->get_name(), indvar_type);
    library_stream << " = " << language_extension_.expression(node_.init()) << ";" << std::endl;

    // Vectorized loop
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    for (auto& local : locals_) {
        if (local == indvar_->get_name()) {
            continue;
        }
        library_stream << this->declaration(local, dynamic_cast<const types::Scalar&>(sdfg_.type(local))) << ";"
                       << std::endl;
    }

    auto update_vec = symbolic::symbol("hn::Lanes(" + daisy_vec(vec_type_) + ")");
    auto condition_vec = symbolic::subs(node_.condition(), indvar_, symbolic::add(indvar_, update_vec));

    library_stream << "for";
    library_stream << "(";
    library_stream << ";";
    library_stream << language_extension_.expression(condition_vec);
    library_stream << ";";
    library_stream << language_extension_.expression(indvar_);
    library_stream << " = ";
    library_stream << language_extension_.expression(symbolic::add(indvar_, update_vec));
    library_stream << ")" << std::endl;

    // Loop body
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    this->dispatch_highway(library_snippet_factory, library_stream, node_.root());

    // End of body
    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl;

    // End of vectorized loop
    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl;

    // Postamble for remaining elements
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    for (auto& local : locals_) {
        if (local == indvar_->get_name()) {
            continue;
        }
        library_stream << this->language_extension_.declaration(local, sdfg_.type(local)) << ";" << std::endl;
    }

    // Loop header
    library_stream << "for";
    library_stream << "(";
    library_stream << ";";
    library_stream << language_extension_.expression(node_.condition());
    library_stream << ";";
    library_stream << language_extension_.expression(indvar_);
    library_stream << " = ";
    library_stream << language_extension_.expression(node_.update());
    library_stream << ")" << std::endl;

    // Postamble loop body
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    codegen::SequenceDispatcher sequence_dispatcher(
        language_extension_, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_
    );
    sequence_dispatcher.dispatch(library_stream, library_stream, library_snippet_factory);

    // End of postamble body
    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl;

    // End of postamble
    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl;
};

void HighwayMapDispatcher::dispatch_node(
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    std::string kernel_name = "highway_kernel_" + std::to_string(node_.element_id());

    // Kernel call and declaration in current file
    this->dispatch_kernel_call(main_stream, kernel_name);

    this->dispatch_kernel_declaration(globals_stream, kernel_name);

    // Kernel definition in separate file

    auto& library_stream = library_snippet_factory.require(kernel_name, "cpp", true).stream();

    std::filesystem::path sdfg_path = sdfg_.metadata("sdfg_file");
    std::filesystem::path kernel_file = sdfg_path.parent_path() / (kernel_name + ".cpp");
    library_stream << "#include " << library_snippet_factory.header_path().filename() << std::endl;
    library_stream << "#undef HWY_TARGET_INCLUDE" << std::endl;
    library_stream << "#define HWY_TARGET_INCLUDE " << "\"" << kernel_file.string() << "\"" << std::endl;
    library_stream << "#include <hwy/foreach_target.h>" << std::endl;
    library_stream << "#include <hwy/highway.h>" << std::endl;
    library_stream << "#include <hwy/contrib/math/math-inl.h>" << std::endl;
    library_stream << std::endl;

    // Begin of namespace
    library_stream << "namespace HWY_NAMESPACE {" << std::endl;
    library_stream << "namespace hn = hwy::HWY_NAMESPACE;" << std::endl << std::endl;

    this->dispatch_kernel_preamble(library_stream, kernel_name);

    // Begin of body
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    this->dispatch_kernel_body(library_snippet_factory, library_stream);

    // End of body
    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl;

    // End of namespace
    library_stream << std::endl;
    library_stream << "}" << std::endl;

    // Dispatch wrapper
    library_stream << std::endl;
    library_stream << "#if HWY_ONCE" << std::endl << std::endl;
    library_stream << "HWY_EXPORT(" << kernel_name << ");" << std::endl;
    library_stream << std::endl;

    library_stream << "extern \"C\" void " << kernel_name;
    library_stream << "(";
    library_stream << helpers::join(arguments_declaration_, ", ");
    library_stream << ")" << std::endl;
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    library_stream << "HWY_DYNAMIC_DISPATCH";
    library_stream << "(" << kernel_name << ")";
    library_stream << "(";
    library_stream << helpers::join(arguments_, ", ");
    library_stream << ");" << std::endl;

    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl << std::endl;
    library_stream << "#endif" << std::endl;
};

void HighwayMapDispatcher::dispatch_highway(
    codegen::CodeSnippetFactory& library_snippet_factory,
    codegen::PrettyPrinter& library_stream,
    structured_control_flow::ControlFlowNode& node
) {
    if (auto block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        this->dispatch_highway(library_snippet_factory, library_stream, *block);
    } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        this->dispatch_highway(library_snippet_factory, library_stream, *sequence);
    } else {
        throw InvalidSDFGException(
            "Schedule type Highway applied on unsupported control flow node: " + std::string(typeid(node).name())
        );
    }
}

void HighwayMapDispatcher::dispatch_highway(
    codegen::CodeSnippetFactory& library_snippet_factory,
    codegen::PrettyPrinter& library_stream,
    structured_control_flow::Sequence& node
) {
    for (size_t i = 0; i < node.size(); i++) {
        if (!node.at(i).second.assignments().empty()) {
            throw InvalidSDFGException("Highway schedule does not support transitions with assignments");
        }

        auto& child = node.at(i).first;
        this->dispatch_highway(library_snippet_factory, library_stream, child);
    }
}

void HighwayMapDispatcher::dispatch_highway(
    codegen::CodeSnippetFactory& library_snippet_factory,
    codegen::PrettyPrinter& library_stream,
    structured_control_flow::Block& node
) {
    auto& arguments_analysis = analysis_manager_.get<analysis::ArgumentsAnalysis>();
    auto locals = arguments_analysis.locals(analysis_manager_, node_);
    symbolic::SymbolSet local_symbols;
    for (auto& local : locals) {
        auto& type = sdfg_.type(local);
        if (types::is_integer(type.primitive_type())) {
            local_symbols.insert(symbolic::symbol(local));
        }
    }

    auto& graph = node.dataflow();
    for (auto& dnode : graph.topological_sort()) {
        if (auto access_node = dynamic_cast<data_flow::AccessNode*>(dnode)) {
            continue;
        }

        library_stream << "{" << std::endl;
        library_stream.setIndent(library_stream.indent() + 4);

        // Declare in connectors
        for (auto& iedge : graph.in_edges(*dnode)) {
            this->dispatch_iedge(library_stream, iedge);
        }

        // Declare out connector
        auto& oedge = *graph.out_edges(*dnode).begin();
        auto& dst_node = static_cast<data_flow::AccessNode&>(oedge.dst());
        library_stream << "auto " << oedge.src_conn() << " = hn::Undefined(";
        library_stream << daisy_vec(oedge.base_type().primitive_type()) << ");" << std::endl;
        library_stream << std::endl;

        // Dispatch code nodes
        if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(dnode)) {
            std::string tasklet_code = HighwayMapDispatcher::tasklet(*tasklet);
            if (tasklet_code.empty()) {
                throw InvalidSDFGException("Schedule type Highway applied on unsupported tasklet node");
            }

            library_stream << tasklet_code << std::endl;
        } else if (auto cmath_node = dynamic_cast<math::cmath::CMathNode*>(dnode)) {
            std::string cmath_code = HighwayMapDispatcher::cmath_node(*cmath_node);
            if (cmath_code.empty()) {
                throw InvalidSDFGException("Schedule type Highway applied on unsupported CMath node");
            }

            library_stream << cmath_code << std::endl;
        } else {
            throw InvalidSDFGException("Schedule type Highway applied on unsupported data flow node");
        }

        library_stream << std::endl;
        if (locals.find(dst_node.data()) != locals.end()) {
            // Local variable
            library_stream << dst_node.data() << " = " << oedge.src_conn() << ";" << std::endl;
        } else {
            // Store to memory
            library_stream << "hn::StoreU(" << oedge.src_conn() << ", ";
            library_stream << daisy_vec(oedge.base_type().primitive_type()) << ", ";
            library_stream << "&";
            library_stream << "(" << this->language_extension_.type_cast(dst_node.data(), oedge.base_type()) << ")";
            library_stream << this->language_extension_.subset(oedge.base_type(), oedge.subset());
            library_stream << ");" << std::endl;
        }

        library_stream.setIndent(library_stream.indent() - 4);
        library_stream << "}" << std::endl;
    }
};

std::string HighwayMapDispatcher::daisy_vec(const types::PrimitiveType& type) {
    switch (type) {
        case types::PrimitiveType::Int8:
            return "daisy_vec_s8";
        case types::PrimitiveType::Int16:
            return "daisy_vec_s16";
        case types::PrimitiveType::Int32:
            return "daisy_vec_s32";
        case types::PrimitiveType::Int64:
            return "daisy_vec_s64";
        case types::PrimitiveType::UInt8:
            return "daisy_vec_u8";
        case types::PrimitiveType::UInt16:
            return "daisy_vec_u16";
        case types::PrimitiveType::UInt32:
            return "daisy_vec_u32";
        case types::PrimitiveType::UInt64:
            return "daisy_vec_u64";
        case types::PrimitiveType::Float:
            return "daisy_vec_f32";
        case types::PrimitiveType::Double:
            return "daisy_vec_f64";
        default:
            throw InvalidSDFGException("Schedule type Highway applied on unsupported types");
    }
}

std::string HighwayMapDispatcher::declaration(const std::string& container, const types::Scalar& type) {
    switch (type.primitive_type()) {
        case types::PrimitiveType::Bool:
            return "auto " + container + " = hn::MaskFalse(" + daisy_vec(types::PrimitiveType::Int8) + ");";
        default:
            return "auto " + container + " = hn::Undefined(" + daisy_vec(type.primitive_type()) + ");";
    }
}

std::string HighwayMapDispatcher::tasklet(data_flow::Tasklet& tasklet) {
    switch (tasklet.code()) {
        case data_flow::TaskletCode::assign:
            return tasklet.output() + " = " + tasklet.input(0) + ";";
        case data_flow::TaskletCode::int_abs:
            return tasklet.output() + " = hn::Abs(" + tasklet.input(0) + ");";
        case data_flow::TaskletCode::int_add:
            return tasklet.output() + " = hn::Add(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_sub:
            return tasklet.output() + " = hn::Sub(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_mul:
            return tasklet.output() + " = hn::Mul(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_sdiv:
            return tasklet.output() + " = hn::Div(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_srem:
            throw std::runtime_error("int_srem not implemented for Highway backend");
        case data_flow::TaskletCode::int_udiv:
            throw std::runtime_error("int_udiv not implemented for Highway backend");
        case data_flow::TaskletCode::int_urem:
            throw std::runtime_error("int_urem not implemented for Highway backend");
        case data_flow::TaskletCode::int_and:
            return tasklet.output() + " = hn::And(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_or:
            return tasklet.output() + " = hn::Or(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_xor:
            return tasklet.output() + " = hn::Xor(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_shl:
            throw std::runtime_error("int_shl not implemented for Highway backend");
        case data_flow::TaskletCode::int_ashr:
            throw std::runtime_error("int_ashr not implemented for Highway backend");
        case data_flow::TaskletCode::int_lshr:
            throw std::runtime_error("int_lshr not implemented for Highway backend");
        case data_flow::TaskletCode::int_smin:
            return tasklet.output() + " = hn::Min(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_smax:
            return tasklet.output() + " = hn::Max(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_umin:
            return tasklet.output() + " = hn::Min(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_scmp:
            throw std::runtime_error("int_scmp not implemented for Highway backend");
        case data_flow::TaskletCode::int_umax:
            return tasklet.output() + " = hn::Max(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_ucmp:
            throw std::runtime_error("int_ucmp not implemented for Highway backend");
        case data_flow::TaskletCode::int_eq:
            return tasklet.output() + " = hn::Eq(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_ne:
            return tasklet.output() + " = hn::Ne(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_sge:
            return tasklet.output() + " = hn::Ge(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_sgt:
            return tasklet.output() + " = hn::Gt(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_sle:
            return tasklet.output() + " = hn::Le(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_slt:
            return tasklet.output() + " = hn::Lt(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_uge:
            return tasklet.output() + " = hn::Ge(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_ugt:
            return tasklet.output() + " = hn::Gt(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_ule:
            return tasklet.output() + " = hn::Le(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::int_ult:
            return tasklet.output() + " = hn::Lt(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_neg:
            return tasklet.output() + " = hn::Neg(" + tasklet.input(0) + ");";
        case data_flow::TaskletCode::fp_add:
            return tasklet.output() + " = hn::Add(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_sub:
            return tasklet.output() + " = hn::Sub(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_mul:
            return tasklet.output() + " = hn::Mul(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_div:
            return tasklet.output() + " = hn::Div(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_rem:
            throw std::runtime_error("fp_rem not implemented for Highway backend");
        case data_flow::TaskletCode::fp_oeq:
            return tasklet.output() + " = hn::Eq(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_one:
            return tasklet.output() + " = hn::Ne(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_oge:
            return tasklet.output() + " = hn::Ge(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_ogt:
            return tasklet.output() + " = hn::Gt(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_ole:
            return tasklet.output() + " = hn::Le(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_olt:
            return tasklet.output() + " = hn::Lt(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_ord:
            throw std::runtime_error("fp_ord not implemented for Highway backend");
        case data_flow::TaskletCode::fp_ueq:
            return tasklet.output() + " = hn::Eq(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_une:
            return tasklet.output() + " = hn::Ne(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_ugt:
            return tasklet.output() + " = hn::Gt(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_uge:
            return tasklet.output() + " = hn::Ge(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_ult:
            return tasklet.output() + " = hn::Lt(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_ule:
            return tasklet.output() + " = hn::Le(" + tasklet.input(0) + ", " + tasklet.input(1) + ");";
        case data_flow::TaskletCode::fp_uno:
            throw std::runtime_error("fp_uno not implemented for Highway backend");
        case data_flow::TaskletCode::fp_fma:
            return tasklet.output() + " = hn::MulAdd(" + tasklet.input(0) + ", " + tasklet.input(1) + ", " +
                   tasklet.input(2) + ");";
        default:
            return "";
    }
};

std::string HighwayMapDispatcher::cmath_node(math::cmath::CMathNode& cmath_node) {
    switch (cmath_node.function()) {
        case math::cmath::CMathFunction::cos:
            return cmath_node.output(0) + " = hn::Cos(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::ceil:
            return cmath_node.output(0) + " = hn::Ceil(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::exp:
            return cmath_node.output(0) + " = hn::Exp(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::exp2:
            return cmath_node.output(0) + " = hn::Exp2(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::fabs:
            return cmath_node.output(0) + " = hn::Abs(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::floor:
            return cmath_node.output(0) + " = hn::Floor(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::log:
            return cmath_node.output(0) + " = hn::Log(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::log2:
            return cmath_node.output(0) + " = hn::Log2(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::log10:
            return cmath_node.output(0) + " = hn::Log10(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::pow:
            return cmath_node.output(0) + " = hn::Pow(" + cmath_node.input(0) + ", " + cmath_node.input(1) + ");";
        case math::cmath::CMathFunction::round:
            return cmath_node.output(0) + " = hn::Round(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::sin:
            return cmath_node.output(0) + " = hn::Sin(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::sqrt:
            return cmath_node.output(0) + " = hn::Sqrt(" + cmath_node.input(0) + ");";
        case math::cmath::CMathFunction::trunc:
            return cmath_node.output(0) + " = hn::Trunc(" + cmath_node.input(0) + ");";
        default:
            return "";
    }
};

void HighwayMapDispatcher::dispatch_iedge(codegen::PrettyPrinter& library_stream, const data_flow::Memlet& memlet) {
    auto& src = static_cast<const data_flow::AccessNode&>(memlet.src());
    auto& base_type = memlet.base_type();

    // Case 1: Constant node
    if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(&src)) {
        std::string data = const_node->data();

        if (base_type.type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException("Schedule type Highway applied on unsupported constant type");
        }
        auto& scalar_type = static_cast<const types::Scalar&>(base_type);

        library_stream << "const auto " << memlet.dst_conn();
        library_stream << " = hn::Set(" << daisy_vec(scalar_type.primitive_type()) << ", ";
        library_stream << data << this->language_extension_.subset(base_type, memlet.subset()) << ");" << std::endl;
        return;
    }

    // Case 2: Local (implicitly scalar)
    if (locals_.find(src.data()) != locals_.end()) {
        library_stream << "const auto " << memlet.dst_conn() << " = ";
        library_stream << src.data() << ";" << std::endl;
        return;
    }

    // Case 3: Argument (any type)
    if (base_type.type_id() == types::TypeID::Scalar) {
        library_stream << "const auto " << memlet.dst_conn() << " = ";
        library_stream << "hn::Set(" << daisy_vec(base_type.primitive_type()) << ", ";
        library_stream << src.data() << ";" << std::endl;
        return;
    } else {
        // Distinguish access type
        auto access_type =
            transformations::HighwayTransform::classify_memlet_access_type(memlet.subset(), indvar_, local_symbols_);

        if (access_type == transformations::HighwayTransform::CONSTANT) {
            library_stream << "const auto " << memlet.dst_conn() << " = ";
            library_stream << "hn::Set(";
            library_stream << daisy_vec(base_type.primitive_type()) << ", ";
            library_stream << "(" << this->language_extension_.type_cast(src.data(), base_type) << ")";
            library_stream << this->language_extension_.subset(base_type, memlet.subset());
            library_stream << ");" << std::endl;
        } else if (access_type == transformations::HighwayTransform::CONTIGUOUS) {
            library_stream << "const auto " << memlet.dst_conn() << " = hn::LoadU(";
            library_stream << daisy_vec(base_type.primitive_type()) << ", ";
            library_stream << "&";
            library_stream << "(" << this->language_extension_.type_cast(src.data(), base_type) << ")";
            library_stream << this->language_extension_.subset(base_type, memlet.subset());
            library_stream << ");" << std::endl;
        } else {
            throw InvalidSDFGException(
                "Schedule type Highway found unsupported memlet access pattern: " + std::to_string(access_type)
            );
        }
        return;
    }
}

codegen::InstrumentationInfo HighwayMapDispatcher::instrumentation_info() const {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    analysis::LoopInfo loop_info = loop_analysis.loop_info(&node_);

    // Perform FlopAnalysis
    std::unordered_map<std::string, std::string> metrics;
    auto& flop_analysis = analysis_manager_.get<analysis::FlopAnalysis>();
    auto flop = flop_analysis.get_if_available_for_codegen(&node_);
    if (!flop.is_null()) {
        std::string flop_str = language_extension_.expression(flop);
        metrics.insert({"flop", flop_str});
    }

    return codegen::InstrumentationInfo(
        node_.element_id(), codegen::ElementType_Map, codegen::TargetType_CPU_PARALLEL, loop_info, metrics
    );
};

} // namespace highway
} // namespace sdfg
