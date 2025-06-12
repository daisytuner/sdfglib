#include "sdfg/element.h"

#include <boost/uuid/uuid_io.hpp>

namespace sdfg {

boost::uuids::random_generator thread_local Element::UUID_GENERATOR;

DebugInfo::DebugInfo()
    : filename_(), start_line_(0), start_column_(0), end_line_(0), end_column_(0), has_(false) {

      };

DebugInfo::DebugInfo(std::string filename, size_t start_line, size_t start_column, size_t end_line,
                     size_t end_column)
    : filename_(filename),
      start_line_(start_line),
      start_column_(start_column),
      end_line_(end_line),
      end_column_(end_column),
      has_(true) {

      };

bool DebugInfo::has() const { return this->has_; };

std::string DebugInfo::filename() const { return this->filename_; };

size_t DebugInfo::start_line() const { return this->start_line_; };

size_t DebugInfo::start_column() const { return this->start_column_; };

size_t DebugInfo::end_line() const { return this->end_line_; };

size_t DebugInfo::end_column() const { return this->end_column_; };

DebugInfo DebugInfo::merge(const DebugInfo& left, const DebugInfo& right) {
    if (!left.has()) {
        return right;
    }
    if (!right.has()) {
        return left;
    }
    if (left.filename() != right.filename()) {
        throw InvalidSDFGException("DebugInfo: Filenames do not match");
    }

    size_t start_line = 0;
    size_t start_column = 0;
    size_t end_line = 0;
    size_t end_column = 0;

    if (left.start_line_ < right.start_line_) {
        start_line = left.start_line_;
        start_column = left.start_column_;
    } else {
        start_line = right.start_line_;
        start_column = right.start_column_;
    }

    if (left.end_line_ > right.end_line_) {
        end_line = left.end_line_;
        end_column = left.end_column_;
    } else {
        end_line = right.end_line_;
        end_column = right.end_column_;
    }
    return DebugInfo(left.filename_, start_line, start_column, end_line, end_column);
};

Element::Element(const DebugInfo& debug_info) : debug_info_(debug_info) {
    boost::uuids::uuid uuid = UUID_GENERATOR();
    this->element_id_ = boost::uuids::to_string(uuid);
};

std::string Element::element_id() const { return this->element_id_; };

const DebugInfo& Element::debug_info() const { return this->debug_info_; };

}  // namespace sdfg
