#pragma once

#include <cassert>
#include <memory>
#include <string>

#include "sdfg/exceptions.h"
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
    size_t start_line_;
    size_t start_column_;
    size_t end_line_;
    size_t end_column_;

    bool has_;
};

class DebugInfoInstruction {
private:
    DebugLoc loc_;

    std::unique_ptr<DebugInfoInstruction> InlinedAt_;

public:
    DebugInfoInstruction();

    DebugInfoInstruction(DebugLoc loc);

    DebugInfoInstruction(DebugLoc loc, std::unique_ptr<DebugInfoInstruction> inlined_at);

    bool has() const;

    std::string filename() const;

    std::string function() const;

    size_t start_line() const;

    size_t start_column() const;

    size_t end_line() const;

    size_t end_column() const;

    static DebugInfoInstruction merge(const DebugInfoInstruction& left, const DebugInfoInstruction& right);
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
