#pragma once

#include <exception>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace transformations {

class Transformation {
   public:
    virtual ~Transformation() = default;

    virtual std::string name() const = 0;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) = 0;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) = 0;

    virtual void to_json(nlohmann::json& j) const { throw std::logic_error("Not implemented"); };
};

class InvalidTransformationException : public std::exception {
   private:
    std::string message_;

   public:
    explicit InvalidTransformationException(const std::string& message) : message_(message) {}

    const char* what() const noexcept override { return message_.c_str(); }
};

class InvalidTransformationDescriptionException : public std::exception {
   private:
    std::string message_;

   public:
    explicit InvalidTransformationDescriptionException(const std::string& message)
        : message_(message) {}

    const char* what() const noexcept override { return message_.c_str(); }
};

}  // namespace transformations
}  // namespace sdfg
