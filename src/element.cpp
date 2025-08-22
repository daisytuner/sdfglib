#include "sdfg/element.h"
#include <string>
#include <vector>
#include "sdfg/exceptions.h"

namespace sdfg {

DebugInfoElement::DebugInfoElement() { this->locations_ = {DebugLoc("", "", 0, 0, false)}; };

DebugInfoElement::DebugInfoElement(DebugLoc loc) : locations_({loc}) {};

DebugInfoElement::DebugInfoElement(DebugLoc loc, std::vector<DebugLoc> inlined_at) : locations_({loc}) {
    for (auto& debug_loc : inlined_at) {
        if (debug_loc.has_) {
            this->locations_.push_back(debug_loc);
        }
    }
};

DebugInfoElement::DebugInfoElement(std::vector<DebugLoc> inlined_at) {
    for (auto& loc : inlined_at) {
        if (loc.has_) {
            this->locations_.push_back(loc);
        }
    }
    if (this->locations_.empty()) {
        this->locations_.push_back(DebugLoc("", "", 0, 0, false));
    }
}

const std::vector<DebugLoc>& DebugInfoElement::locations() const { return this->locations_; }

bool DebugInfoElement::has() const { return this->locations_.front().has_; };

std::string DebugInfoElement::filename() const { return this->locations_.front().filename_; };

std::string DebugInfoElement::function() const { return this->locations_.front().function_; };

size_t DebugInfoElement::line() const { return this->locations_.front().line_; };

size_t DebugInfoElement::column() const { return this->locations_.front().column_; };

DebugInfo::DebugInfo()
    : instructions_(), filename_(""), function_(""), start_line_(0), start_column_(0), end_line_(0), end_column_(0),
      has_(false) {};

DebugInfo::DebugInfo(DebugInfoElement loc) {
    if (loc.has()) {
        this->instructions_.push_back(loc);
        this->filename_ = loc.filename();
        this->function_ = loc.function();
        this->start_line_ = loc.line();
        this->start_column_ = loc.column();
        this->end_line_ = loc.line();
        this->end_column_ = loc.column();
        this->has_ = true;
        instructions_.push_back(std::move(loc));
    } else {
        this->has_ = false;
    }
};

bool DebugInfo::has() const { return has_; }

std::string DebugInfo::filename() const { return filename_; }

std::string DebugInfo::function() const { return function_; }

size_t DebugInfo::start_line() const { return start_line_; }

size_t DebugInfo::start_column() const { return start_column_; }

size_t DebugInfo::end_line() const { return end_line_; }

size_t DebugInfo::end_column() const { return end_column_; }

const std::vector<DebugInfoElement>& DebugInfo::instructions() const { return this->instructions_; };

bool fuse_ranges(
    const DebugLoc& loc,
    std::string& filename,
    std::string& function,
    size_t& line_start,
    size_t& col_start,
    size_t& line_end,
    size_t& col_end
) {
    if (loc.filename_ == filename && loc.function_ == function) {
        if (line_start == loc.line_) {
            col_start = std::min(col_start, loc.column_);
        } else if (line_start > loc.line_) {
            col_start = loc.column_;
        }
        line_start = std::min(line_start, loc.line_);

        if (line_end == loc.line_) {
            col_end = std::max(col_end, loc.column_);
        } else if (line_end < loc.line_) {
            col_end = loc.column_;
        }
        line_end = std::max(line_end, loc.line_);
        return true; // Same file and function, ranges fused
    }
    return false; // Different file or function, ranges not fused
}

DebugInfo::DebugInfo(std::vector<DebugInfoElement> instructions) {
    if (instructions.empty()) {
        this->has_ = false;
        return;
    }

    this->has_ = false;
    for (const auto& instruction : instructions) {
        if (instruction.has()) {
            this->has_ = true;
            this->instructions_.push_back(instruction);
        }
    }

    if (!this->has_) {
        return; // No valid instructions found
    }

    // find locations
    std::string filename;
    std::string function;
    size_t start_line;
    size_t start_column;
    size_t end_line;
    size_t end_column;


    bool found = false;
    for (auto& loc : this->instructions_.front().locations()) {
        filename = loc.filename_;
        function = loc.function_;
        start_line = loc.line_;
        start_column = loc.column_;
        end_line = loc.line_;
        end_column = loc.column_;
        int fitting = 0;
        for (const auto& instruction : this->instructions_) {
            found = false;
            for (const auto& inlined_loc : instruction.locations()) {
                if (fuse_ranges(inlined_loc, filename, function, start_line, start_column, end_line, end_column)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                // If no ranges were fused, we can break early
                break;
            }
            fitting++;
        }
        if (fitting == this->instructions_.size()) {
            found = true;
            break; // All instructions fit the same file and function
        }
    }
    if (!found) {
        throw InvalidSDFGException("No valid debug locations found in DebugInfo");
    }

    this->filename_ = filename;
    this->function_ = function;
    this->start_line_ = start_line;
    this->start_column_ = start_column;
    this->end_line_ = end_line;
    this->end_column_ = end_column;
};

DebugInfo DebugInfo::merge(const DebugInfo& left, const DebugInfo& right) {
    if (left.has() && !right.has()) {
        return left; // If left has debug info, return it
    }
    if (!left.has() && right.has()) {
        return right; // If right has debug info, return it
    }

    auto list = left.instructions();
    list.insert(list.end(), right.instructions().begin(), right.instructions().end());
    return DebugInfo(list);
};

void DebugInfo::append(DebugInfoElement& other) {
    if (!other.has()) {
        return;
    }

    if (!this->has_) {
        this->instructions_.push_back(other);
        this->filename_ = other.filename();
        this->function_ = other.function();
        this->start_line_ = other.line();
        this->start_column_ = other.column();
        this->end_line_ = other.line();
        this->end_column_ = other.column();
        this->has_ = true;
        return;
    }

    this->instructions_.push_back(other);

    DebugInfo debug_info(this->instructions_);

    this->filename_ = debug_info.filename();
    this->function_ = debug_info.function();
    this->start_line_ = debug_info.start_line();
    this->start_column_ = debug_info.start_column();
    this->end_line_ = debug_info.end_line();
    this->end_column_ = debug_info.end_column();
    this->has_ = true;
};

Element::Element(size_t element_id, const DebugInfo& debug_info) : element_id_(element_id), debug_info_(debug_info) {};

size_t Element::element_id() const { return this->element_id_; };

const DebugInfo& Element::debug_info() const { return this->debug_info_; };

void Element::set_debug_info(const DebugInfo& debug_info) { this->debug_info_ = debug_info; }

} // namespace sdfg
