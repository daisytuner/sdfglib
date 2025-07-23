#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

namespace sdfg {
namespace math {
namespace blas {

enum BLAS_Precision {
    h,
    s,
    d,
    c,
    z,
};

enum BLAS_Transpose {
    No = 111,
    Trans = 112,
    ConjTrans = 113,
};

enum BLAS_Layout {
    RowMajor = 101,
    ColMajor = 102,
};

inline data_flow::ImplementationType ImplementationType_BLAS{"BLAS"};
inline data_flow::ImplementationType ImplementationType_CUBLAS{"CUBLAS"};

} // namespace blas
} // namespace math
} // namespace sdfg
