#pragma once

#include <cstddef>
#include <unordered_map>
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/map.h"


namespace sdfg {
namespace codegen {


typedef StringEnum ElementType;
inline ElementType ElementType_Map{"map"};
inline ElementType ElementType_For{"for"};
inline ElementType ElementType_While{"while"};
inline ElementType ElementType_Block{"block"};
inline ElementType ElementType_IfElse{"if_else"};
inline ElementType ElementType_Sequence{"sequence"};
inline ElementType ElementType_H2DTransfer{"h2d_transfer"};
inline ElementType ElementType_D2HTransfer{"d2h_transfer"};
inline ElementType ElementType_Unknown{"unknown"};

typedef StringEnum TargetType;
inline TargetType TargetType_SEQUENTIAL{structured_control_flow::ScheduleType_Sequential::value()};
inline TargetType TargetType_CPU_PARALLEL{structured_control_flow::ScheduleType_CPU_Parallel::value()};

class InstrumentationInfo {
private:
    ElementType element_type_;
    TargetType target_type_;
    long long loopnest_index_;
    size_t element_id_;

    std::unordered_map<std::string, std::string> metrics_;

public:
    InstrumentationInfo(
        const ElementType& element_type,
        const TargetType& target_type,
        long long loopnest_index,
        size_t element_id,
        const std::unordered_map<std::string, std::string>& metrics = {}
    );

    const ElementType& element_type() const;

    const TargetType& target_type() const;

    long long loopnest_index() const;

    size_t element_id() const;

    const std::unordered_map<std::string, std::string>& metrics() const;
};

} // namespace codegen
} // namespace sdfg
