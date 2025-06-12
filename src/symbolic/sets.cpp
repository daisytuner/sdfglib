#include "sdfg/symbolic/sets.h"

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>

namespace sdfg {
namespace symbolic {

bool intersect(const MultiExpression& expr1, const SymbolSet& params1, const MultiExpression& expr2,
               const SymbolSet& params2, const Assumptions& assums) {
    return false;
}

}  // namespace symbolic
}  // namespace sdfg