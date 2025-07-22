#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

namespace sdfg {
namespace math {
namespace blas {

enum class BLAS_Precision {
    h,
    s,
    d,
    c,
    z,
};

enum class BLAS_Transpose {
    No,
    Trans,
    ConjTrans,
};

enum class BLAS_Layout {
    RowMajor,
    ColMajor,
};


} // namespace blas
} // namespace math
} // namespace sdfg
