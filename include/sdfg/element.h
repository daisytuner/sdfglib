#pragma once

#include <cassert>
#include <string>
#include <vector>

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

struct DebugLoc {
    std::string filename_;
    std::string function_;
    size_t line_;
    size_t column_;

    bool has_;
};

class DebugInfoElement {
private:
    std::vector<DebugLoc> locations_;

public:
    DebugInfoElement();

    DebugInfoElement(DebugLoc loc);

    DebugInfoElement(DebugLoc loc, std::vector<DebugLoc> inlined_at);

    bool has() const;

    std::string filename() const;

    std::string function() const;

    size_t line() const;

    size_t column() const;

    const std::vector<DebugLoc>& locations() const;
};

class DebugInfo {
private:
    std::vector<DebugInfoElement> instructions_;

    std::string filename_;

    std::string function_;

    size_t line_start_;

    size_t column_start_;

    size_t line_end_;

    size_t column_end_;

    bool has_;

public:
    DebugInfo();

    DebugInfo(DebugInfoElement loc);

    DebugInfo(std::vector<DebugInfoElement> instructions);

    DebugInfo(const DebugInfo& other);

    const std::vector<DebugInfoElement>& instructions() const;

    DebugInfo merge(DebugInfo& left, DebugInfo& right);

    DebugInfo& append(DebugInfoElement& other);
};

class Element {
    friend class builder::SDFGBuilder;
    friend class builder::StructuredSDFGBuilder;
    friend class serializer::JSONSerializer;

protected:
    size_t element_id_;
    DebugInfo debug_info_;

public:
    Element(size_t element_id, const DebugInfo& debug_info);

    virtual ~Element() = default;

    size_t element_id() const;

    const DebugInfo& debug_info() const;

    void set_debug_info(const DebugInfo& debug_info);

    /**
     * Validates the element.
     *
     * @throw InvalidSDFGException if the element is invalid
     */
    virtual void validate(const Function& function) const = 0;

    virtual void replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) = 0;
};

} // namespace sdfg
