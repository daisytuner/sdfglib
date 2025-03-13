#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

bool Transformation::can_be_applied(Schedule& schedule) {
    throw std::logic_error("Not implemented");
};

void Transformation::apply(Schedule& schedule) { throw std::logic_error("Not implemented"); };

bool Transformation::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                    analysis::AnalysisManager& analysis_manager) {
    throw std::logic_error("Not implemented");
};

void Transformation::apply(builder::StructuredSDFGBuilder& builder,
                           analysis::AnalysisManager& analysis_manager) {
    throw std::logic_error("Not implemented");
};

}  // namespace transformations
}  // namespace sdfg
