#pragma once

#include <string>

#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace cuda {

inline std::string CUDA_DEVICE_PREFIX = "__daisy_cuda_";

enum CUDADimension { X = 0, Y = 1, Z = 2 };

class ScheduleType_CUDA {
public:
    static void dimension(structured_control_flow::ScheduleType& schedule, const CUDADimension& dimension);
    static CUDADimension dimension(const structured_control_flow::ScheduleType& schedule);
    static void block_size(structured_control_flow::ScheduleType& schedule, const symbolic::Expression block_size);
    static symbolic::Integer block_size(const structured_control_flow::ScheduleType& schedule);
    static bool nested_sync(const structured_control_flow::ScheduleType& schedule);
    static void nested_sync(structured_control_flow::ScheduleType& schedule, const bool nested_sync);
    static const std::string value() { return "CUDA"; }
    static structured_control_flow::ScheduleType create() {
        auto schedule_type = structured_control_flow::ScheduleType(value());
        dimension(schedule_type, CUDADimension::X);
        return schedule_type;
    }
};

inline codegen::TargetType TargetType_CUDA{ScheduleType_CUDA::value()};

} // namespace cuda
} // namespace sdfg
