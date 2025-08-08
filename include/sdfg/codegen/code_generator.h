#pragma once

#include <filesystem>
#include <sstream>
#include <string>

#include "code_snippet_factory.h"
#include "sdfg/analysis/mem_access_range_analysis.h"
#include "sdfg/codegen/instrumentation/capture_var_plan.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/type.h"

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
    InstrumentationPlan& instrumentation_plan_;

    /// @brief Stream for includes
    PrettyPrinter includes_stream_;

    /// @brief Stream for classes
    PrettyPrinter classes_stream_;

    /// @brief Stream for global variables and functions
    PrettyPrinter globals_stream_;

    /// @brief Stream for library functions
    CodeSnippetFactory library_snippet_factory_;

    /// @brief Main stream
    PrettyPrinter main_stream_;

    /// @brief Emit instrumentation code to capture runtime contents of inputs and outputs
    bool capture_args_results_;

    std::tuple<int, types::PrimitiveType> analyze_type_rec(
        symbolic::Expression* curr_dim,
        int max_dim,
        int dim_idx,
        const types::IType& type,
        int arg_idx,
        const analysis::MemAccessRange* range,
        analysis::AnalysisManager& analysis_manager,
        const StructuredSDFG& sdfg,
        std::string var_name
    );

    bool add_capture_plan(
        const std::string& name,
        int argIdx,
        bool isExternal,
        std::vector<CaptureVarPlan>& plan,
        const analysis::MemAccessRanges& ranges
    );

public:
    CodeGenerator(
        StructuredSDFG& sdfg,
        InstrumentationPlan& instrumentation_plan,
        bool capture_args_results = false,
        const std::pair<std::filesystem::path, std::filesystem::path>* output_and_header_paths = nullptr
    )
        : sdfg_(sdfg), instrumentation_plan_(instrumentation_plan), library_snippet_factory_(output_and_header_paths),
          capture_args_results_(capture_args_results) {};


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
    virtual bool as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) = 0;

    /// @brief Generate only the function source code and append it to the source file. @ref as_source generates this
    /// into `source_path` after includes, globals and structs
    virtual void append_function_source(std::ofstream& ofs_source) = 0;

    /// @brief Get the includes
    const PrettyPrinter& includes() const { return this->includes_stream_; };

    /// @brief Get the classes
    const PrettyPrinter& classes() const { return this->classes_stream_; };

    /// @brief Get the globals
    const PrettyPrinter& globals() const { return this->globals_stream_; };

    /// @brief all created library snippets
    const std::unordered_map<std::string, CodeSnippet>& library_snippets() const {
        return this->library_snippet_factory_.snippets();
    };

    /// @brief Get the main stream
    const PrettyPrinter& main() const { return this->main_stream_; };

    std::unique_ptr<std::vector<CaptureVarPlan>> create_capture_plans();
};

} // namespace codegen
} // namespace sdfg
