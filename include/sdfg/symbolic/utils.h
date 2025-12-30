/**
 * @file utils.h
 * @brief ISL (Integer Set Library) integration utilities
 *
 * This file provides utilities for converting symbolic expressions to ISL format
 * and performing operations using the Integer Set Library. ISL is used for:
 * - Representing integer sets and relations
 * - Computing set operations (union, intersection, difference)
 * - Checking set properties (emptiness, disjointness, subset relations)
 * - Analyzing iteration spaces and memory access patterns
 *
 * ## ISL Integration
 *
 * The Integer Set Library (ISL) is a powerful tool for reasoning about integer sets
 * and relations. This module provides the bridge between sdfglib's symbolic expressions
 * and ISL's representation, enabling sophisticated polyhedral analysis.
 *
 * Key functions include:
 * - Converting expressions to ISL map strings
 * - Generating constraints from assumptions
 * - Canonicalizing map dimensions
 * - Delinearizing multi-dimensional expressions
 *
 * ## Example Usage
 *
 * @code
 * // Convert a memory access pattern to ISL format
 * auto i = symbolic::symbol("i");
 * auto j = symbolic::symbol("j");
 * MultiExpression expr = {i, j};  // 2D access pattern
 * 
 * Assumptions assums;
 * assums[i].add_lower_bound(symbolic::zero());
 * assums[i].add_upper_bound(symbolic::integer(10));
 * assums[j].add_lower_bound(symbolic::zero());
 * assums[j].add_upper_bound(symbolic::integer(20));
 * 
 * std::string isl_str = expression_to_map_str(expr, assums);
 * // ISL string can be used with ISL library functions
 * @endcode
 *
 * @see sets.h for high-level set operations using ISL
 * @see assumptions.h for symbol assumptions used in constraints
 */

#pragma once

#include <isl/map.h>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

/**
 * @brief Converts a multi-dimensional expression to an ISL map string
 * @param expr Multi-dimensional expression representing iteration or access space
 * @param assums Assumptions about symbol bounds
 * @return ISL-formatted string representing the expression as a map
 *
 * Converts symbolic expressions to ISL's string format for use with ISL library
 * functions. The resulting string includes constraints from assumptions.
 */
std::string expression_to_map_str(const MultiExpression& expr, const Assumptions& assums);

/**
 * @brief Converts two expressions to ISL format for intersection analysis
 * @param expr1 First multi-dimensional expression
 * @param expr2 Second multi-dimensional expression
 * @param indvar Induction variable for the iteration space
 * @param assums1 Assumptions for first expression
 * @param assums2 Assumptions for second expression
 * @return Tuple of (map1_str, map2_str, combined_constraints_str) in ISL format
 *
 * Prepares two expressions and their assumptions for intersection checking using ISL.
 * Returns formatted strings that can be parsed by ISL to construct maps and perform
 * set operations.
 */
std::tuple<std::string, std::string, std::string> expressions_to_intersection_map_str(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
);

/**
 * @brief Generates constraint expressions from assumptions
 * @param syms Set of symbols to generate constraints for
 * @param assums Assumptions containing bounds information
 * @param seen Set of symbols already processed (to avoid duplicates)
 * @return Set of constraint expressions derived from assumptions
 *
 * Extracts bound constraints from assumptions and converts them to constraint
 * expressions suitable for ISL. This includes lower bounds (sym >= lb) and
 * upper bounds (sym <= ub) for all symbols.
 */
ExpressionSet generate_constraints(SymbolSet& syms, const Assumptions& assums, SymbolSet& seen);

/**
 * @brief Converts a constraint expression to ISL string format
 * @param con Constraint expression (typically a comparison)
 * @return ISL-formatted string for the constraint
 *
 * Converts individual constraint expressions (like "i >= 0" or "i < N") to
 * ISL's string representation.
 */
std::string constraint_to_isl_str(const Expression con);

/**
 * @brief Canonicalizes dimension names in an ISL map
 * @param map ISL map to modify
 * @param in_prefix Prefix for input dimensions (e.g., "i")
 * @param out_prefix Prefix for output dimensions (e.g., "o")
 *
 * Renames the dimensions of an ISL map to use canonical names with specified prefixes.
 * This ensures consistent naming across different maps for easier composition and
 * comparison.
 */
void canonicalize_map_dims(isl_map* map, const std::string& in_prefix, const std::string& out_prefix);

/**
 * @brief Delinearizes a multi-dimensional expression
 * @param expr Multi-dimensional expression potentially containing linearized indices
 * @param assums Assumptions about symbols
 * @return Delinearized multi-dimensional expression
 *
 * Attempts to recover multi-dimensional structure from linearized expressions.
 * For example, if an expression represents a linearized 2D array access A[i*N + j],
 * this function tries to recover the 2D indices [i, j].
 */
MultiExpression delinearize(const MultiExpression& expr, const Assumptions& assums);

} // namespace symbolic
} // namespace sdfg
