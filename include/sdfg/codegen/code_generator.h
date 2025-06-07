#pragma once

#include <filesystem>
#include <sstream>
#include <string>

#include "sdfg/codegen/instrumentation/instrumentation_strategy.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace codegen {

/**
 * @brief Base class for code generators
 *
 * This class provides the basic structure for code generators.
 * It contains the streams for the includes, classes, globals, library functions and main code.
 */
class CodeGenerator {
   protected:
    /// @brief Reference to the schedule
    StructuredSDFG& sdfg_;

    /// @brief Instrumentation strategy
    InstrumentationStrategy instrumentation_strategy_;

    /// @brief Stream for includes
    PrettyPrinter includes_stream_;

    /// @brief Stream for classes
    PrettyPrinter classes_stream_;

    /// @brief Stream for global variables and functions
    PrettyPrinter globals_stream_;

    /// @brief Stream for library functions
    PrettyPrinter library_stream_;

    /// @brief Main stream
    PrettyPrinter main_stream_;

   public:
    CodeGenerator(StructuredSDFG& sdfg, InstrumentationStrategy instrumentation_strategy)
        : sdfg_(sdfg), instrumentation_strategy_(instrumentation_strategy) {};

    virtual ~CodeGenerator() = default;

    /**
     * @brief Generate the code
     *
     * @return true if the code was generated successfully
     */
    virtual bool generate() = 0;

    /// @brief Generate a function definition for the SDFG
    virtual std::string function_definition() = 0;

    /// @brief Generate the SDFG's code into source files
    virtual bool as_source(const std::filesystem::path& header_path,
                           const std::filesystem::path& source_path,
                           const std::filesystem::path& library_path) = 0;

    /// @brief Get the includes
    const PrettyPrinter& includes() const { return this->includes_stream_; };

    /// @brief Get the classes
    const PrettyPrinter& classes() const { return this->classes_stream_; };

    /// @brief Get the globals
    const PrettyPrinter& globals() const { return this->globals_stream_; };

    /// @brief Get the library
    const PrettyPrinter& library() const { return this->library_stream_; };

    /// @brief Get the main stream
    const PrettyPrinter& main() const { return this->main_stream_; };
};

}  // namespace codegen
}  // namespace sdfg