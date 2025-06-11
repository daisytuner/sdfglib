#include "sdfg/codegen/dispatchers/library_nodes/thread_barrier_dispatcher.h"

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"

namespace sdfg {
namespace codegen {

void ThreadBarrierDispatcher::dispatch(PrettyPrinter& stream) {
    if (dynamic_cast<CLanguageExtension*>(&this->language_extension_) != nullptr) {
        throw std::runtime_error(
            "ThreadBarrierDispatcher is not supported for C language extension. Use CUDA language "
            "extension instead.");
    } else if (dynamic_cast<CPPLanguageExtension*>(&this->language_extension_) != nullptr) {
        throw std::runtime_error(
            "ThreadBarrierDispatcher is not supported for C++ language extension. Use CUDA "
            "language extension instead.");
    } else if (dynamic_cast<CUDALanguageExtension*>(&this->language_extension_) != nullptr) {
        stream << "__syncthreads();" << std::endl;
    } else {
        throw std::runtime_error("Unsupported language extension for ThreadBarrierDispatcher");
    }
}

}  // namespace codegen
}  // namespace sdfg