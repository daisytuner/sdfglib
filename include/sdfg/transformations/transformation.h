#pragma once

#include <exception>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/optimization_report/pass_report_consumer.h"

namespace sdfg {

class PassReportConsumer;

namespace transformations {

class Transformation {
public:
    virtual ~Transformation() = default;

    virtual void set_report(PassReportConsumer* report) {}

    virtual std::string name() const = 0;

    // TODO builder and probably analysis manager should be given via constructor, as in practice, the nodes which are
    // transformed are already constructor params, and therefore the SDFG is implied
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) = 0;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) = 0;

    /**
     * Bundles [can_be_applied] and [apply] into a single function so they can share state
     * @return true if was applied
     */
    virtual bool try_apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
        auto can_be = can_be_applied(builder, analysis_manager);

        if (!can_be) {
            return false;
        } else {
            apply(builder, analysis_manager);
            return true;
        }
    }

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
    explicit InvalidTransformationDescriptionException(const std::string& message) : message_(message) {}

    const char* what() const noexcept override { return message_.c_str(); }
};

} // namespace transformations
} // namespace sdfg
