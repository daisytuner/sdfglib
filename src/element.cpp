#include "sdfg/element.h"
#include <string>
#include "sdfg/exceptions.h"

namespace sdfg {

DebugInfoElement::DebugInfoElement() { this->locations_ = {DebugLoc("", "", 0, 0, false)}; };

DebugInfoElement::DebugInfoElement(DebugLoc loc)
    : locations_({loc}) {

      };

DebugInfoElement::DebugInfoElement(DebugLoc loc, std::vector<DebugLoc> inlined_at) : locations_({loc}) {
    for (auto& loc : inlined_at) {
        if (loc.has_) {
            this->locations_.push_back(loc);
        }
    }
};


bool DebugInfoElement::has() const { return this->locations_.front().has_; };

std::string DebugInfoElement::filename() const { return this->locations_.front().filename_; };

std::string DebugInfoElement::function() const { return this->locations_.front().function_; };

size_t DebugInfoElement::line() const { return this->locations_.front().line_; };

size_t DebugInfoElement::column() const { return this->locations_.front().column_; };

const std::vector<DebugInfoElement>& DebugInfo::instructions() const { return this->instructions_; };

DebugInfo::DebugInfo()
    : instructions_(), filename_(""), function_(""), line_start_(0), column_start_(0), line_end_(0), column_end_(0),
      has_(false) {};

DebugInfo::DebugInfo(DebugInfoElement loc) : instructions_() {
    if (loc.has()) {
        this->instructions_.push_back(loc);
        this->filename_ = loc.filename();
        this->function_ = loc.function();
        this->line_start_ = loc.line();
        this->column_start_ = loc.column();
        this->line_end_ = loc.line();
        this->column_end_ = loc.column();
        this->has_ = true;
        instructions_.push_back(std::move(loc));
    } else {
        this->has_ = false;
    }
};

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
    size_t line_start;
    size_t col_start;
    size_t line_end;
    size_t col_end;


    bool found = false;
    for (auto& loc : this->instructions_.front().locations()) {
        filename = loc.filename_;
        function = loc.function_;
        line_start = loc.line_;
        col_start = loc.column_;
        line_end = loc.line_;
        col_end = loc.column_;
        for (const auto& instruction : this->instructions_) {
            for (const auto& inlined_loc : instruction.locations()) {
                if (fuse_ranges(inlined_loc, filename, function, line_start, col_start, line_end, col_end)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                // If no ranges were fused, we can break early
                break;
            }
        }
    }
    if (!found) {
        throw InvalidSDFGException("No valid debug locations found in DebugInfo");
    }

    this->filename_ = filename;
    this->function_ = function;
    this->line_start_ = line_start;
    this->column_start_ = col_start;
    this->line_end_ = line_end;
    this->column_end_ = col_end;
};

DebugInfo DebugInfo::merge(DebugInfo left, DebugInfo right) {
    // TODO: Implement merging logic
};

DebugInfo& DebugInfo::append(DebugInfoElement& other) {
    // TODO: Implement append logic
};

Element::Element(size_t element_id, const DebugInfo& debug_info) : element_id_(element_id), debug_info_(debug_info) {};

size_t Element::element_id() const { return this->element_id_; };

const DebugInfo& Element::debug_info() const { return this->debug_info_; };

void Element::set_debug_info(const DebugInfo& debug_info) { this->debug_info_ = debug_info; }

} // namespace sdfg
