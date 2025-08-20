#include "sdfg/element.h"

namespace sdfg {

DebugInfo::DebugInfo()
    : loc_("", "", 0, 0, 0, 0, false), InlinedAt_(nullptr) {

      };

DebugInfo::DebugInfo(DebugLoc loc)
    : loc_(loc), InlinedAt_(nullptr) {

      };

DebugInfo::DebugInfo(DebugLoc loc, std::unique_ptr<DebugInfo> inlined_at)
    : loc_(loc), InlinedAt_(std::move(inlined_at)) {

      };


bool DebugInfo::has() const { return this->loc_.has_; };

std::string DebugInfo::filename() const { return this->loc_.filename_; };

std::string DebugInfo::function() const { return this->loc_.function_; };

size_t DebugInfo::start_line() const { return this->loc_.start_line_; };

size_t DebugInfo::start_column() const { return this->loc_.start_column_; };

size_t DebugInfo::end_line() const { return this->loc_.end_line_; };

size_t DebugInfo::end_column() const { return this->loc_.end_column_; };

DebugInfo DebugInfo::merge(const DebugInfo& left, const DebugInfo& right) {
    if (!left.has()) {
        return right;
    }
    if (!right.has()) {
        return left;
    }
    if (left.filename() != right.filename()) {
        return left;
    }
    if (left.function() != right.function() && left.function() != "" && right.function() != "") {
        return left;
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

    std::string function = left.function_;
    if (left.function_ == "") {
        function = right.function_;
    }

    return DebugInfo(left.filename_, function, start_line, start_column, end_line, end_column);
};

Element::Element(size_t element_id, const DebugInfo& debug_info) : element_id_(element_id), debug_info_(debug_info) {};

size_t Element::element_id() const { return this->element_id_; };

const DebugInfo& Element::debug_info() const { return this->debug_info_; };

void Element::set_debug_info(const DebugInfo& debug_info) { this->debug_info_ = debug_info; };

} // namespace sdfg
