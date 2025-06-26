#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace sdfg {
namespace helpers {

// Transformation functions
template <typename T>
inline T& indirect(const std::unique_ptr<T>& ptr) {
    return *ptr;
};

template <typename T>
inline const T& add_const(T& s) {
    return s;
};

// string operations

inline bool is_number(const std::string& s) {
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
};

inline uint64_t parse_number(const std::string& s) { return std::stoull(s); };

inline bool endswith(std::string const& value, std::string const& ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
};

template <class S, class T>
inline std::string join(std::vector<T>& elems, S& delim) {
    if (elems.size() == 0) {
        return "";
    }
    std::stringstream ss;
    typename std::vector<T>::iterator e = elems.begin();
    ss << *e++;
    for (; e != elems.end(); ++e) {
        ss << delim << *e;
    }
    return ss.str();
};

inline void split(std::vector<std::string>& result, std::string s, std::string del = " ") {
    int start, end = -1 * del.size();
    do {
        start = end + del.size();
        end = s.find(del, start);
        result.push_back(s.substr(start, end - start));
    } while (end != -1);
}

template <typename T>
inline bool sets_intersect(const std::unordered_set<T>& set1, const std::unordered_set<T>& set2) {
    // Determine the smaller and larger set
    const std::unordered_set<T>& smaller = (set1.size() < set2.size()) ? set1 : set2;
    const std::unordered_set<T>& larger = (set1.size() < set2.size()) ? set2 : set1;

    // Iterate through the smaller set and check for existence in the larger set
    for (const T& element : smaller) {
        if (larger.find(element) != larger.end()) {
            return true;  // Intersection found
        }
    }

    return false;  // No intersection found
}

}  // namespace helpers
}  // namespace sdfg
