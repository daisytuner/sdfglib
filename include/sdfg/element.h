#pragma once

#include <cassert>
#include <cstddef>
#include <string>

#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace serializer {
class JSONSerializer;
} // namespace serializer

class DebugInfo {
private:
    std::string filename;
    size_t start_line;
    size_t start_column;
    size_t end_line;
    size_t end_column;

    bool has;

public:
    DebugInfo();

    DebugInfo(std::string filename, size_t start_line, size_t start_column, size_t end_line, size_t end_column);

    [[nodiscard]] auto has() const -> bool;

    [[nodiscard]] auto filename() const -> std::string;

    [[nodiscard]] auto start_line() const -> size_t;

    [[nodiscard]] auto start_column() const -> size_t;

    [[nodiscard]] auto end_line() const -> size_t;

    [[nodiscard]] auto end_column() const -> size_t;

    static auto merge(const DebugInfo& left, const DebugInfo& right) -> DebugInfo;
};

class Element {
    friend class builder::SDFGBuilder;
    friend class builder::StructuredSDFGBuilder;
    friend class serializer::JSONSerializer;

protected:
    size_t element_id;
    DebugInfo debug_info;

public:
    Element(size_t element_id, const DebugInfo& debug_info);

    virtual ~Element() = default;

    [[nodiscard]] auto element_id() const -> size_t;

    [[nodiscard]] auto debug_info() const -> const DebugInfo&;

    virtual void replace(const symbolic::Expression& old_expression, const symbolic::Expression& new_expression) = 0;
};

} // namespace sdfg
