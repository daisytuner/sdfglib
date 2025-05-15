#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/structure.h"
#include "sdfg/types/type.h"

namespace sdfg {

namespace data_flow {

typedef std::vector<symbolic::Expression> Subset;

}

namespace types {

const types::IType& infer_type(const sdfg::Function& function, const types::IType& type,
                               const data_flow::Subset& subset);

std::unique_ptr<types::IType> recombine_array_type(const types::IType& type, uint depth,
                                                   const types::IType& inner_type);

}  // namespace types
}  // namespace sdfg
