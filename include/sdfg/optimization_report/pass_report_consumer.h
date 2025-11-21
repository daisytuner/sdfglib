#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {

namespace transformations {
class Transformation;
}

class PassReportConsumer {
public:
    virtual void transform_impossible(const transformations::Transformation* transform, const std::string& reason) = 0;

    virtual void transform_possible(const transformations::Transformation* transform) = 0;

    virtual void transform_applied(const sdfg::transformations::Transformation* transform) = 0;

    virtual void in_scope(StructuredSDFG* scope) = 0;
    void no_scope() { in_scope(nullptr); }

    virtual void in_outermost_loop(int idx) = 0;

    void no_loop() { in_outermost_loop(-1); }
    virtual ~PassReportConsumer() = default;

    virtual void target_transform_possible(const std::string basicString, bool b) = 0;
};

} // namespace sdfg
