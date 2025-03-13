#include "sdfg/codegen/dispatchers/kernel_dispatcher.h"

#include "sdfg/codegen/language_extension.h"
#include "sdfg/schedule.h"

namespace sdfg {
namespace codegen {

KernelDispatcher::KernelDispatcher(LanguageExtension& language_extension, Schedule& schedule,
                                   structured_control_flow::Kernel& node, bool instrumented)
    : NodeDispatcher(language_extension, schedule, node, instrumented),
      node_(node){

      };

void KernelDispatcher::dispatch_node(PrettyPrinter& main_stream, PrettyPrinter& globals_stream,
                                     PrettyPrinter& library_stream) {
    bool gridDim_x = false;
    bool gridDim_y = false;
    bool gridDim_z = false;
    bool blockDim_x = false;
    bool blockDim_y = false;
    bool blockDim_z = false;
    bool blockIdx_x = false;
    bool blockIdx_y = false;
    bool blockIdx_z = false;
    bool threadIdx_x = false;
    bool threadIdx_y = false;
    bool threadIdx_z = false;

    for (auto& container : schedule_.sdfg().containers()) {
        if (container == node_.gridDim_x()->get_name()) {
            gridDim_x = true;
        } else if (container == node_.gridDim_y()->get_name()) {
            gridDim_y = true;
        } else if (container == node_.gridDim_z()->get_name()) {
            gridDim_z = true;
        } else if (container == node_.blockDim_x()->get_name()) {
            blockDim_x = true;
        } else if (container == node_.blockDim_y()->get_name()) {
            blockDim_y = true;
        } else if (container == node_.blockDim_z()->get_name()) {
            blockDim_z = true;
        } else if (container == node_.blockIdx_x()->get_name()) {
            blockIdx_x = true;
        } else if (container == node_.blockIdx_y()->get_name()) {
            blockIdx_y = true;
        } else if (container == node_.blockIdx_z()->get_name()) {
            blockIdx_z = true;
        } else if (container == node_.threadIdx_x()->get_name()) {
            threadIdx_x = true;
        } else if (container == node_.threadIdx_y()->get_name()) {
            threadIdx_y = true;
        } else if (container == node_.threadIdx_z()->get_name()) {
            threadIdx_z = true;
        }
    }

    if (gridDim_x) {
        main_stream << node_.gridDim_x()->get_name() << " = "
                    << language_extension_.expression(node_.gridDim_x_init()) << ";" << std::endl;
    }
    if (gridDim_y) {
        main_stream << node_.gridDim_y()->get_name() << " = "
                    << language_extension_.expression(node_.gridDim_y_init()) << ";" << std::endl;
    }
    if (gridDim_z) {
        main_stream << node_.gridDim_z()->get_name() << " = "
                    << language_extension_.expression(node_.gridDim_z_init()) << ";" << std::endl;
    }
    if (blockDim_x) {
        main_stream << node_.blockDim_x()->get_name() << " = "
                    << language_extension_.expression(node_.blockDim_x_init()) << ";" << std::endl;
    }
    if (blockDim_y) {
        main_stream << node_.blockDim_y()->get_name() << " = "
                    << language_extension_.expression(node_.blockDim_y_init()) << ";" << std::endl;
    }
    if (blockDim_z) {
        main_stream << node_.blockDim_z()->get_name() << " = "
                    << language_extension_.expression(node_.blockDim_z_init()) << ";" << std::endl;
    }
    if (blockIdx_x) {
        main_stream << node_.blockIdx_x()->get_name() << " = "
                    << language_extension_.expression(node_.blockIdx_x_init()) << ";" << std::endl;
    }
    if (blockIdx_y) {
        main_stream << node_.blockIdx_y()->get_name() << " = "
                    << language_extension_.expression(node_.blockIdx_y_init()) << ";" << std::endl;
    }
    if (blockIdx_z) {
        main_stream << node_.blockIdx_z()->get_name() << " = "
                    << language_extension_.expression(node_.blockIdx_z_init()) << ";" << std::endl;
    }
    if (threadIdx_x) {
        main_stream << node_.threadIdx_x()->get_name() << " = "
                    << language_extension_.expression(node_.threadIdx_x_init()) << ";" << std::endl;
    }
    if (threadIdx_y) {
        main_stream << node_.threadIdx_y()->get_name() << " = "
                    << language_extension_.expression(node_.threadIdx_y_init()) << ";" << std::endl;
    }
    if (threadIdx_z) {
        main_stream << node_.threadIdx_z()->get_name() << " = "
                    << language_extension_.expression(node_.threadIdx_z_init()) << ";" << std::endl;
    }

    main_stream << "{" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    SequenceDispatcher dispatcher(language_extension_, schedule_, node_.root(), false);
    dispatcher.dispatch(main_stream, globals_stream, library_stream);

    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;
};

}  // namespace codegen
}  // namespace sdfg
