#pragma once

#include <cassert>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

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

    DebugInfoElement(std::vector<DebugLoc> inlined_at);

    bool has() const;

    std::string filename() const;

    std::string function() const;

    size_t line() const;

    size_t column() const;

    const std::vector<DebugLoc>& locations() const;
};

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
    DebugInfoRegion(std::unordered_set<size_t> indices, const std::vector<DebugInfoElement>& all_instructions);

    std::unordered_set<size_t> indices() const;

    static DebugInfoRegion
    merge(DebugInfoRegion& first, DebugInfoRegion& second, const std::vector<DebugInfoElement>& all_instructions);

    bool has() const;

    std::string filename() const;
    std::string function() const;
    size_t start_line() const;
    size_t start_column() const;
    size_t end_line() const;
    size_t end_column() const;
};

class DebugInfo {
private:
    std::vector<DebugInfoElement> instructions_;

public:
    size_t add_element(DebugInfoElement loc);

    DebugInfoRegion get_region(std::unordered_set<size_t> indices) const;

    const std::vector<DebugInfoElement>& instructions() const;
};


} // namespace sdfg
