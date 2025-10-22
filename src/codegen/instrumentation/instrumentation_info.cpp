#include "sdfg/codegen/instrumentation/instrumentation_info.h"


namespace sdfg {
namespace codegen {


InstrumentationInfo::InstrumentationInfo(
    const ElementType& element_type,
    const TargetType& target_type,
    long long loopnest_index,
    size_t element_id,
    const std::unordered_map<std::string, std::string>& metrics
)
    : element_type_(element_type), target_type_(target_type), loopnest_index_(loopnest_index), element_id_(element_id),
      metrics_(metrics) {}

const ElementType& InstrumentationInfo::element_type() const { return element_type_; }

const TargetType& InstrumentationInfo::target_type() const { return target_type_; }

long long InstrumentationInfo::loopnest_index() const { return loopnest_index_; }

size_t InstrumentationInfo::element_id() const { return element_id_; }

const std::unordered_map<std::string, std::string>& InstrumentationInfo::metrics() const { return metrics_; }

} // namespace codegen
} // namespace sdfg
