#pragma once

#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
namespace sdfg {
namespace cuda {
namespace blas {

void create_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension);

void destroy_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension);

void cublas_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
);

} // namespace blas
} // namespace cuda
} // namespace sdfg
