#include "sdfg/codegen/dispatchers/map_dispatcher.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace codegen {

MapDispatcher::MapDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

void MapDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    auto dispatcher = MapDispatcherRegistry::instance().get_map_dispatcher(node_.schedule_type().value());
    if (dispatcher) {
        auto dispatcher_ptr =
            dispatcher(language_extension_, sdfg_, analysis_manager_, node_, instrumentation_plan_, arg_capture_plan_);
        dispatcher_ptr->dispatch_node(main_stream, globals_stream, library_snippet_factory);
    } else {
        throw std::runtime_error("Unsupported map schedule type: " + std::string(node_.schedule_type().value()));
    }
};

SequentialMapDispatcher::SequentialMapDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

void SequentialMapDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    main_stream << "// Map" << std::endl;
    main_stream << "for";
    main_stream << "(";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.init());
    main_stream << ";";
    main_stream << language_extension_.expression(node_.condition());
    main_stream << ";";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.update());
    main_stream << ")" << std::endl;
    main_stream << "{" << std::endl;

    main_stream.setIndent(main_stream.indent() + 4);
    SequenceDispatcher
        dispatcher(language_extension_, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

CPUParallelMapDispatcher::CPUParallelMapDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

void CPUParallelMapDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    // Mark written locals as private
    analysis::AnalysisManager analysis_manager(sdfg_);
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView users_view(users, node_.root());

    std::vector<std::string> locals;
    for (auto& entry : users.locals(node_.root())) {
        if (users_view.writes(entry).size() > 0 || users_view.moves(entry).size() > 0) {
            locals.push_back(entry);
        }
    }

    // Generate code
    main_stream << "// Map" << std::endl;
    main_stream << "#pragma omp parallel for";

    main_stream << " schedule(";
    if (structured_control_flow::ScheduleType_CPU_Parallel::omp_schedule(node_.schedule_type()) ==
        structured_control_flow::OpenMPSchedule::Static) {
        main_stream << "static)";
    } else if (structured_control_flow::ScheduleType_CPU_Parallel::omp_schedule(node_.schedule_type()) ==
               structured_control_flow::OpenMPSchedule::Dynamic) {
        main_stream << "dynamic)";
    } else if (structured_control_flow::ScheduleType_CPU_Parallel::omp_schedule(node_.schedule_type()) ==
               structured_control_flow::OpenMPSchedule::Guided) {
        main_stream << "guided)";
    } else {
        throw std::runtime_error("Unsupported OpenMP schedule type");
    }

    if (!structured_control_flow::ScheduleType_CPU_Parallel::num_threads(node_.schedule_type()).is_null()) {
        main_stream << " num_threads(";
        main_stream
            << language_extension_
                   .expression(structured_control_flow::ScheduleType_CPU_Parallel::num_threads(node_.schedule_type()));
        main_stream << ")";
    }

    if (locals.size() > 0) {
        main_stream << " private(" << helpers::join(locals, ", ") << ")";
    }
    main_stream << std::endl;
    main_stream << "for";
    main_stream << "(";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.init());
    main_stream << ";";
    main_stream << language_extension_.expression(node_.condition());
    main_stream << ";";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.update());
    main_stream << ")" << std::endl;
    main_stream << "{" << std::endl;

    for (auto& local : locals) {
        auto& type = sdfg_.type(local);
        if (type.storage_type().allocation() == types::StorageType::AllocationType::Managed) {
            if (type.storage_type().is_cpu_stack()) {
                main_stream << local << " = ";
                main_stream << "alloca(" << language_extension_.expression(type.storage_type().allocation_size())
                            << ")";
                main_stream << ";" << std::endl;
            } else if (type.storage_type().is_cpu_heap()) {
                main_stream << local << " = ";
                main_stream << language_extension_.external_prefix() << "malloc("
                            << language_extension_.expression(type.storage_type().allocation_size()) << ")";
                main_stream << ";" << std::endl;
            }
        }
    }

    main_stream.setIndent(main_stream.indent() + 4);
    SequenceDispatcher
        dispatcher(language_extension_, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
    main_stream.setIndent(main_stream.indent() - 4);

    for (auto& local : locals) {
        auto& type = sdfg_.type(local);
        if (type.storage_type().deallocation() == types::StorageType::AllocationType::Managed) {
            if (type.storage_type().is_cpu_heap()) {
                main_stream << language_extension_.external_prefix() << "free(" << local << ");" << std::endl;
            }
        }
    }

    main_stream << "}" << std::endl;
};

InstrumentationInfo MapDispatcher::instrumentation_info() const {
    auto dispatcher = MapDispatcherRegistry::instance().get_map_dispatcher(node_.schedule_type().value());
    if (dispatcher) {
        auto dispatcher_ptr =
            dispatcher(language_extension_, sdfg_, analysis_manager_, node_, instrumentation_plan_, arg_capture_plan_);
        auto map_dispatcher_ptr = static_cast<MapDispatcher*>(dispatcher_ptr.get());
        return map_dispatcher_ptr->instrumentation_info();
    } else {
        throw std::runtime_error("Unsupported map schedule type: " + std::string(node_.schedule_type().value()));
    }
};

InstrumentationInfo SequentialMapDispatcher::instrumentation_info() const {
    size_t loopnest_index = -1;
    auto& loop_tree_analysis = analysis_manager_.get<analysis::LoopAnalysis>();

    auto outermost_loops = loop_tree_analysis.outermost_loops();
    for (size_t i = 0; i < outermost_loops.size(); i++) {
        if (outermost_loops[i] == &node_) {
            loopnest_index = i;
            break;
        }
    }

    // Perform FlopAnalysis
    std::unordered_map<std::string, std::string> metrics;
    auto& flop_analysis = analysis_manager_.get<analysis::FlopAnalysis>();
    if (flop_analysis.contains(&node_)) {
        auto flop = flop_analysis.get(&node_);
        if (!flop.is_null()) {
            if (!symbolic::has_dynamic_sizeof(flop)) {
                std::string flop_str = language_extension_.expression(flop);
                metrics.insert({"flop", flop_str});
            }
        }
    }

    return InstrumentationInfo(ElementType_Map, TargetType_SEQUENTIAL, loopnest_index, node_.element_id(), metrics);
};

InstrumentationInfo CPUParallelMapDispatcher::instrumentation_info() const {
    size_t loopnest_index = -1;
    auto& loop_tree_analysis = analysis_manager_.get<analysis::LoopAnalysis>();

    auto outermost_loops = loop_tree_analysis.outermost_loops();
    for (size_t i = 0; i < outermost_loops.size(); i++) {
        if (outermost_loops[i] == &node_) {
            loopnest_index = i;
            break;
        }
    }

    // Perform FlopAnalysis
    std::unordered_map<std::string, std::string> metrics;
    auto& flop_analysis = analysis_manager_.get<analysis::FlopAnalysis>();
    if (flop_analysis.contains(&node_)) {
        auto flop = flop_analysis.get(&node_);
        if (!flop.is_null()) {
            if (!symbolic::has_dynamic_sizeof(flop)) {
                std::string flop_str = language_extension_.expression(flop);
                metrics.insert({"flop", flop_str});
            }
        }
    }

    return InstrumentationInfo(ElementType_Map, TargetType_CPU_PARALLEL, loopnest_index, node_.element_id(), metrics);
};

} // namespace codegen
} // namespace sdfg
