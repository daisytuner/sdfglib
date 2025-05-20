#pragma once

#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"

namespace sdfg {
namespace codegen {

class CPPCodeGenerator : public CodeGenerator {
   private:
    CPPLanguageExtension language_extension_;

   protected:
    void dispatch_includes();

    void dispatch_structures();

    void dispatch_globals();

    void dispatch_schedule();

   public:
    CPPCodeGenerator(ConditionalSchedule& schedule);
    CPPCodeGenerator(ConditionalSchedule& schedule, InstrumentationStrategy instrumentation_strategy);

    bool generate() override;

    std::string function_definition() override;

    bool as_source(const std::filesystem::path& header_path,
                   const std::filesystem::path& source_path,
                   const std::filesystem::path& library_path) override;
};

}  // namespace codegen
}  // namespace sdfg
