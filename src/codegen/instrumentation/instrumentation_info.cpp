#include "sdfg/codegen/instrumentation/instrumentation_info.h"


namespace sdfg {
namespace codegen {


InstrumentationInfo::InstrumentationInfo(
    size_t element_id,
    const ElementType& element_type,
    const TargetType& target_type,
    const analysis::LoopInfo& loop_info,
    const std::unordered_map<std::string, std::string>& metrics
)
    : element_id_(element_id), element_type_(element_type), target_type_(target_type), loop_info_(loop_info),
      metrics_(metrics) {}

size_t InstrumentationInfo::element_id() const { return element_id_; }

const ElementType& InstrumentationInfo::element_type() const { return element_type_; }

const TargetType& InstrumentationInfo::target_type() const { return target_type_; }

const analysis::LoopInfo& InstrumentationInfo::loop_info() const { return loop_info_; }

const std::unordered_map<std::string, std::string>& InstrumentationInfo::metrics() const { return metrics_; }

} // namespace codegen
} // namespace sdfg
