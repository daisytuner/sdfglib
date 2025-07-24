#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

namespace sdfg {
namespace math {
namespace blas {

enum BLAS_Precision {
    h = 'h',
    s = 's',
    d = 'd',
    c = 'c',
    z = 'z',
};

constexpr std::string_view BLAS_Precision_to_string(BLAS_Precision precision) {
    switch (precision) {
        case BLAS_Precision::h:
            return "h";
        case BLAS_Precision::s:
            return "s";
        case BLAS_Precision::d:
            return "d";
        case BLAS_Precision::c:
            return "c";
        case BLAS_Precision::z:
            return "z";
        default:
            throw std::runtime_error("Invalid BLAS_Precision value");
    }
}

enum BLAS_Transpose {
    No = 111,
    Trans = 112,
    ConjTrans = 113,
};

constexpr std::string_view BLAS_Transpose_to_string(BLAS_Transpose transpose) {
    switch (transpose) {
        case BLAS_Transpose::No:
            return "CblasNoTrans";
        case BLAS_Transpose::Trans:
            return "CblasTrans";
        case BLAS_Transpose::ConjTrans:
            return "CblasConjTrans";
        default:
            throw std::runtime_error("Invalid BLAS_Transpose value");
    }
}

inline constexpr char BLAS_Transpose_to_char(BLAS_Transpose transpose) {
    switch (transpose) {
        case BLAS_Transpose::No:
            return 'N';
        case BLAS_Transpose::Trans:
            return 'T';
        case BLAS_Transpose::ConjTrans:
            return 'C';
        default:
            throw std::runtime_error("Invalid BLAS_Transpose value");
    }
}

enum BLAS_Layout {
    RowMajor = 101,
    ColMajor = 102,
};

constexpr std::string_view BLAS_Layout_to_string(BLAS_Layout layout) {
    switch (layout) {
        case BLAS_Layout::RowMajor:
            return "CblasRowMajor";
        case BLAS_Layout::ColMajor:
            return "CblasColMajor";
    }
}

inline constexpr std::string_view BLAS_Layout_to_short_string(BLAS_Layout layout) {
    switch (layout) {
        case BLAS_Layout::RowMajor:
            return "RowM";
        case BLAS_Layout::ColMajor:
            return "ColM";
        default:
            throw std::runtime_error("Invalid BLAS_Layout value");
    }
}

inline data_flow::ImplementationType ImplementationType_BLAS{"BLAS"};
inline data_flow::ImplementationType ImplementationType_CUBLAS{"CUBLAS"};

} // namespace blas
} // namespace math
} // namespace sdfg
