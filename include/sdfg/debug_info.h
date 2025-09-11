#pragma once

#include <cassert>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

namespace sdfg {

struct DebugLoc {
    std::string filename;
    std::string function;
    size_t line;
    size_t column;

    bool has;

    bool operator==(const DebugLoc& other) const {
        return this->filename == other.filename && this->function == other.function && this->line == other.line &&
               this->column == other.column && this->has == other.has;
    }
};

class DebugInfo {
private:
    std::vector<DebugLoc> locations_;

public:
    DebugInfo();

    DebugInfo(DebugLoc loc);

    DebugInfo(DebugLoc loc, std::vector<DebugLoc> inlined_at);

    DebugInfo(std::vector<DebugLoc> inlined_at);

    bool has() const;

    std::string filename() const;

    std::string function() const;

    size_t line() const;

    size_t column() const;

    const std::vector<DebugLoc>& locations() const;

    bool operator==(const DebugInfo& other) const;
};

typedef std::vector<DebugInfo> DebugInfos;

class DebugInfoRegion {
private:
    std::unordered_set<size_t> indices_;

    std::string filename_;
    std::string function_;
    size_t start_line_;
    size_t start_column_;
    size_t end_line_;
    size_t end_column_;

    bool has_;

    static bool fuse_ranges(
        const DebugLoc& loc,
        std::string& filename,
        std::string& function,
        size_t& line_start,
        size_t& col_start,
        size_t& line_end,
        size_t& col_end
    );

public:
    DebugInfoRegion();

    DebugInfoRegion(std::unordered_set<size_t> indices, const DebugInfos& all_instructions);

    std::unordered_set<size_t> indices() const;

    static DebugInfoRegion
    merge(const DebugInfoRegion& first, const DebugInfoRegion& second, const DebugInfos& all_instructions);

    bool has() const;

    std::string filename() const;
    std::string function() const;
    size_t start_line() const;
    size_t start_column() const;
    size_t end_line() const;
    size_t end_column() const;
};

class DebugTable {
private:
    DebugInfos elements_;

public:
    size_t add_element(DebugInfo element);

    DebugInfos get_region(std::unordered_set<size_t> indices) const;

    const DebugInfos& elements() const;
};

} // namespace sdfg
