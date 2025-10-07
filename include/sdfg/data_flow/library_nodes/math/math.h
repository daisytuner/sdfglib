#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"

// Intrinsics (math.h)
#include "sdfg/data_flow/library_nodes/math/intrinsic.h"

// BLAS
#include "sdfg/data_flow/library_nodes/math/blas/blas.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm.h"

// ML
#include "sdfg/data_flow/library_nodes/math/ml/abs.h"
#include "sdfg/data_flow/library_nodes/math/ml/add.h"
#include "sdfg/data_flow/library_nodes/math/ml/div.h"
#include "sdfg/data_flow/library_nodes/math/ml/elu.h"
#include "sdfg/data_flow/library_nodes/math/ml/erf.h"
#include "sdfg/data_flow/library_nodes/math/ml/hard_sigmoid.h"
#include "sdfg/data_flow/library_nodes/math/ml/leaky_relu.h"
#include "sdfg/data_flow/library_nodes/math/ml/mul.h"
#include "sdfg/data_flow/library_nodes/math/ml/pow.h"
#include "sdfg/data_flow/library_nodes/math/ml/relu.h"
#include "sdfg/data_flow/library_nodes/math/ml/sigmoid.h"
#include "sdfg/data_flow/library_nodes/math/ml/sqrt.h"
#include "sdfg/data_flow/library_nodes/math/ml/sub.h"
#include "sdfg/data_flow/library_nodes/math/ml/tanh.h"
