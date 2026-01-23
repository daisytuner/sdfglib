#include "sdfg/targets/cuda/blas/utils.h"

#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
namespace sdfg {
namespace cuda {
namespace blas {

void create_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension) {
    stream << "cublasHandle_t handle;" << std::endl;
    stream << "cublasStatus_t status_create = cublasCreate(&handle);" << std::endl;
    cublas_error_checking(stream, language_extension, "status_create");
}

void destroy_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension) {
    stream << "cublasStatus_t status_destroy = cublasDestroy(handle);" << std::endl;
    cublas_error_checking(stream, language_extension, "status_destroy");
}

void cublas_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
) {
    stream << "if (" << status_variable << " != CUBLAS_STATUS_SUCCESS) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << language_extension.external_prefix() << "fprintf(stderr, \"CUBLAS error: %d File: %s, Line: %d\\n\", "
           << status_variable << ", __FILE__, __LINE__);" << std::endl;
    stream << language_extension.external_prefix() << "exit(EXIT_FAILURE);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace blas
} // namespace cuda
} // namespace sdfg
