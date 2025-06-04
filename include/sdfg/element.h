#pragma once

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <cassert>
#include <string>

#include "sdfg/exceptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
}  // namespace builder

class DebugInfo {
   private:
    std::string filename_;
    size_t start_line_;
    size_t start_column_;
    size_t end_line_;
    size_t end_column_;

    bool has_;

   public:
    DebugInfo();

    DebugInfo(std::string filename, size_t start_line, size_t start_column, size_t end_line,
              size_t end_column);

    bool has() const;

    std::string filename() const;

    size_t start_line() const;

    size_t start_column() const;

    size_t end_line() const;

    size_t end_column() const;

    static DebugInfo merge(const DebugInfo& left, const DebugInfo& right);
};

class Element {
    friend class builder::SDFGBuilder;
    friend class builder::StructuredSDFGBuilder;

    static thread_local boost::uuids::random_generator UUID_GENERATOR;

   protected:
    std::string element_id_;
    DebugInfo debug_info_;

   public:
    Element(const DebugInfo& debug_info);

    virtual ~Element() = default;

    std::string element_id() const;

    const DebugInfo& debug_info() const;

    virtual void replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) = 0;
};

}  // namespace sdfg