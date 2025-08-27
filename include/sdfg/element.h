#pragma once

#include <cassert>

#include "sdfg/debug_info.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace serializer {
class JSONSerializer;
} // namespace serializer

class Function;

class Element {
    friend class builder::SDFGBuilder;
    friend class builder::StructuredSDFGBuilder;
    friend class serializer::JSONSerializer;

protected:
    size_t element_id_;
    DebugInfoRegion debug_info_;

public:
    Element(size_t element_id, const DebugInfoRegion& debug_info);

    virtual ~Element() = default;

    size_t element_id() const;

    const DebugInfoRegion& debug_info() const;

    void set_debug_info(const DebugInfoRegion& debug_info);

    /**
     * Validates the element.
     *
     * @throw InvalidSDFGException if the element is invalid
     */
    virtual void validate(const Function& function) const = 0;

    virtual void replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) = 0;
};

} // namespace sdfg
