#include "sdfg/debug_info.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#include "sdfg/exceptions.h"

namespace sdfg {

/***** DebugInfo *****/

DebugInfo::DebugInfo() { this->locations_ = {DebugLoc("", "", 0, 0, false)}; };

DebugInfo::DebugInfo(DebugLoc loc) : locations_({loc}) {};

DebugInfo::DebugInfo(DebugLoc loc, std::vector<DebugLoc> inlined_at) : locations_({loc}) {
    for (auto& debug_loc : inlined_at) {
        if (debug_loc.has) {
            this->locations_.push_back(debug_loc);
        }
    }
};

DebugInfo::DebugInfo(std::vector<DebugLoc> inlined_at) {
    for (auto& loc : inlined_at) {
        if (loc.has) {
            this->locations_.push_back(loc);
        }
    }
    if (this->locations_.empty()) {
        this->locations_.push_back(DebugLoc("", "", 0, 0, false));
    }
}

const std::vector<DebugLoc>& DebugInfo::locations() const { return this->locations_; }

bool DebugInfo::has() const { return this->locations_.front().has; };

std::string DebugInfo::filename() const { return this->locations_.front().filename; };

std::string DebugInfo::function() const { return this->locations_.front().function; };

size_t DebugInfo::line() const { return this->locations_.front().line; };

size_t DebugInfo::column() const { return this->locations_.front().column; };

bool DebugInfo::operator==(const DebugInfo& other) const {
    bool result = true;
    if (this->locations_.size() != other.locations_.size()) return false;
    for (size_t i = 0; i < this->locations_.size(); i++) {
        result &= (this->locations_[i] == other.locations_[i]);
    }
    return result;
}


/***** DebugInfoRegion *****/

DebugInfoRegion::DebugInfoRegion()
    : indices_(), filename_(""), function_(""), start_line_(0), start_column_(0), end_line_(0), end_column_(0) {
    this->has_ = false;
};

DebugInfoRegion::DebugInfoRegion(std::unordered_set<size_t> indices, const DebugInfos& all_instructions)
    : indices_(indices) {
    if (this->indices_.empty()) {
        this->has_ = false;
        return;
    }

    if (all_instructions.empty()) {
        this->has_ = false;
        return;
    }

    // filter instructions
    DebugInfos instructions;
    for (auto index : this->indices_) {
        if (index < all_instructions.size()) {
            auto instruction = all_instructions[index];
            if (instruction.has()) {
                instructions.push_back(instruction);
            }
        }
    }

    if (instructions.empty()) {
        this->has_ = false;
        return;
    }

    // find locations
    std::string filename;
    std::string function;
    size_t start_line;
    size_t start_column;
    size_t end_line;
    size_t end_column;


    bool found = false;
    for (auto& loc : instructions.front().locations()) {
        filename = loc.filename;
        function = loc.function;
        start_line = loc.line;
        start_column = loc.column;
        end_line = loc.line;
        end_column = loc.column;
        int fitting = 0;
        for (const auto& instruction : instructions) {
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
        if (fitting == instructions.size()) {
            found = true;
            break; // All instructions fit the same file and function
        }
    }
    if (!found) {
        throw InvalidSDFGException("No valid debug locations found in DebugTable");
    }

    this->filename_ = filename;
    this->function_ = function;
    this->start_line_ = start_line;
    this->start_column_ = start_column;
    this->end_line_ = end_line;
    this->end_column_ = end_column;
    this->has_ = true;
};

bool DebugInfoRegion::fuse_ranges(
    const DebugLoc& loc,
    std::string& filename,
    std::string& function,
    size_t& line_start,
    size_t& col_start,
    size_t& line_end,
    size_t& col_end
) {
    if (loc.filename == filename && loc.function == function) {
        if (line_start == loc.line) {
            col_start = std::min(col_start, loc.column);
        } else if (line_start > loc.line) {
            col_start = loc.column;
        }
        line_start = std::min(line_start, loc.line);

        if (line_end == loc.line) {
            col_end = std::max(col_end, loc.column);
        } else if (line_end < loc.line) {
            col_end = loc.column;
        }
        line_end = std::max(line_end, loc.line);
        return true; // Same file and function, ranges fused
    }
    return false; // Different file or function, ranges not fused
}

std::unordered_set<size_t> DebugInfoRegion::indices() const { return this->indices_; }

DebugInfoRegion DebugInfoRegion::
    merge(const DebugInfoRegion& first, const DebugInfoRegion& second, const DebugInfos& all_instructions) {
    // Merge the two regions by combining their indices
    std::unordered_set<size_t> merged_indices = first.indices_;
    merged_indices.insert(second.indices_.begin(), second.indices_.end());

    // Create a new region with the merged indices
    return DebugInfoRegion(merged_indices, all_instructions);
}

bool DebugInfoRegion::has() const { return this->has_; }

std::string DebugInfoRegion::filename() const { return this->filename_; }

std::string DebugInfoRegion::function() const { return this->function_; }

size_t DebugInfoRegion::start_line() const { return this->start_line_; }
size_t DebugInfoRegion::start_column() const { return this->start_column_; }
size_t DebugInfoRegion::end_line() const { return this->end_line_; }
size_t DebugInfoRegion::end_column() const { return this->end_column_; }

/***** DebugTable *****/

const DebugInfos& DebugTable::elements() const { return this->elements_; };

size_t DebugTable::add_element(DebugInfo element) {
    auto entry = std::find(this->elements_.begin(), this->elements_.end(), element);
    if (entry != this->elements_.end()) {
        // Element already exists, return its index
        return std::distance(this->elements_.begin(), entry);
    }
    this->elements_.push_back(element);
    return this->elements_.size() - 1;
};

DebugInfos DebugTable::get_region(std::unordered_set<size_t> indices) const {
    DebugInfos elements;
    for (auto index : indices) {
        if (index < this->elements_.size()) {
            elements.push_back(this->elements_[index]);
        }
    }
    return elements;
};

} // namespace sdfg
