#include "sdfg/codegen/dispatchers/map_dispatcher.h"

#include "sdfg/analysis/users.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/tribool.h"

namespace sdfg {
namespace codegen {

MapDispatcher::MapDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::Map& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation_plan), node_(node) {

      };

void MapDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    auto dispatcher = MapDispatcherRegistry::instance().get_map_dispatcher(node_.schedule_type().value());
    if (dispatcher) {
        auto dispatcher_ptr = dispatcher(language_extension_, sdfg_, node_, instrumentation_plan_);
        dispatcher_ptr->dispatch_node(main_stream, globals_stream, library_snippet_factory);
    } else {
        throw std::runtime_error("Unsupported map schedule type: " + std::string(node_.schedule_type().value()));
    }
};

SequentialMapDispatcher::SequentialMapDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::Map& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation_plan), node_(node) {

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
    SequenceDispatcher dispatcher(language_extension_, sdfg_, node_.root(), instrumentation_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

CPUParallelMapDispatcher::CPUParallelMapDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    structured_control_flow::Map& node,
    InstrumentationPlan& instrumentation_plan
)
    : NodeDispatcher(language_extension, sdfg, node, instrumentation_plan), node_(node) {

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

    bool print_parallel = true;
    if (structured_control_flow::ScheduleType_CPU_Parallel::num_threads(node_.schedule_type()) != SymEngine::null) {
        if (symbolic::
                eq(structured_control_flow::ScheduleType_CPU_Parallel::num_threads(node_.schedule_type()),
                   symbolic::one())) {
            print_parallel = false;
        }
    }

    // Generate code
    main_stream << "// Map" << std::endl;
    bool tasking = structured_control_flow::ScheduleType_CPU_Parallel::tasking(node_.schedule_type());
    if (print_parallel) {
        if (tasking) {
            main_stream << "#pragma omp parallel";
            if (structured_control_flow::ScheduleType_CPU_Parallel::num_threads(node_.schedule_type()) !=
                SymEngine::null) {
                main_stream << " num_threads(";
                main_stream << language_extension_.expression(structured_control_flow::ScheduleType_CPU_Parallel::
                                                                  num_threads(node_.schedule_type()));
                main_stream << ")";
            }
            main_stream << std::endl;
            main_stream << "{" << std::endl;
            main_stream << "#pragma omp for";
        } else {
            main_stream << "#pragma omp parallel for";
            if (structured_control_flow::ScheduleType_CPU_Parallel::num_threads(node_.schedule_type()) !=
                SymEngine::null) {
                main_stream << " num_threads(";
                main_stream << language_extension_.expression(structured_control_flow::ScheduleType_CPU_Parallel::
                                                                  num_threads(node_.schedule_type()));
                main_stream << ")";
            }
        }


        main_stream << " schedule(";
        if (structured_control_flow::ScheduleType_CPU_Parallel::omp_schedule(node_.schedule_type()) ==
            structured_control_flow::OpenMPSchedule::Static) {
            main_stream << "static";
        } else if (structured_control_flow::ScheduleType_CPU_Parallel::omp_schedule(node_.schedule_type()) ==
                   structured_control_flow::OpenMPSchedule::Dynamic) {
            main_stream << "dynamic";
        } else if (structured_control_flow::ScheduleType_CPU_Parallel::omp_schedule(node_.schedule_type()) ==
                   structured_control_flow::OpenMPSchedule::Guided) {
            main_stream << "guided";
        } else {
            throw std::runtime_error("Unsupported OpenMP schedule type");
        }

        auto chunk_size = structured_control_flow::ScheduleType_CPU_Parallel::chunk_size(node_.schedule_type());
        if (chunk_size != SymEngine::null) {
            main_stream << ", ";
            main_stream << language_extension_.expression(chunk_size);
        }

        main_stream << ")";

        if (locals.size() > 0) {
            main_stream << " private(" << helpers::join(locals, ", ") << ")";
        }
    } else {
        main_stream << "//Note: map scheduled as CPU_Parallel, but num_threads is 1.";
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

    if (print_parallel && tasking) {
        main_stream << "#pragma omp task" << std::endl;
        main_stream << "{" << std::endl;
    }

    main_stream.setIndent(main_stream.indent() + 4);
    SequenceDispatcher dispatcher(language_extension_, sdfg_, node_.root(), instrumentation_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
    main_stream.setIndent(main_stream.indent() - 4);

    if (print_parallel && tasking) {
        main_stream << "}" << std::endl;
    }

    main_stream << "}" << std::endl;

    if (print_parallel && tasking) {
        main_stream << "#pragma omp taskwait" << std::endl;
        main_stream << "}" << std::endl;
    }
};

} // namespace codegen
} // namespace sdfg
