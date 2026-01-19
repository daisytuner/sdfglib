#include <llvm-19/llvm/ADT/ArrayRef.h>
#include <llvm-19/llvm/ADT/SmallVector.h>
#include <llvm-19/llvm/Support/LogicalResult.h>
#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"

// SDFG Dialect Operations
#define GET_OP_CLASSES
#include "mlir/Dialect/SDFG/IR/SDFGOps.cpp.inc"

// SDFG Dialect Enum Attributes
#include "mlir/Dialect/SDFG/IR/SDFGOpsEnums.cpp.inc"

namespace mlir {
namespace sdfg {

llvm::LogicalResult TaskletOp::verify() {
    size_t code_arity = arity(this->getCode());
    size_t num_operands = this->getNumOperands();
    if (code_arity != num_operands) {
        return this->emitError() << "expects " << code_arity << " operands, but got " << num_operands;
    }
    return success();
}

void SDFGOp::print(OpAsmPrinter& p) {
    bool is_external = this->getBody().empty();
    p << " ";

    // Print visibility
    if (auto visibility = this->getSymVisibilityAttr()) {
        p << visibility.getValue() << " ";
    }

    // Print name as symbol
    p.printSymbolName(this->getSymName());

    // Print function signature: Arguments
    p << " (";
    ArrayRef<Type> arg_types = this->getFunctionType().getInputs();
    for (size_t i = 0, e = arg_types.size(); i < e; ++i) {
        if (i > 0) {
            p << ", ";
        }
        if (is_external) {
            p.printType(arg_types[i]);
        } else {
            p.printRegionArgument(this->getBody().getArgument(i));
        }
    }
    p << ")";

    // Print function signature: Result type
    ArrayRef<Type> result_types = this->getFunctionType().getResults();
    if (!result_types.empty()) {
        assert(result_types.size() == 1);
        p << " -> ";
        p.printType(result_types.front());
    }

    // Print optional function body
    if (!is_external) {
        p << " ";
        p.printRegion(this->getBody(), false, true, true);
    }
}

ParseResult SDFGOp::parse(OpAsmParser& parser, OperationState& result) {
    auto& builder = parser.getBuilder();

    // Parse visibility
    (void) impl::parseOptionalVisibilityKeyword(parser, result.attributes);

    // Parse name as symbol
    StringAttr nameAttr;
    if (parser.parseSymbolName(nameAttr, "sym_name", result.attributes)) {
        return failure();
    }

    // Parse function signature: Arguments
    SMLoc signature_loc = parser.getCurrentLocation();
    llvm::SmallVector<OpAsmParser::Argument> entry_args;
    auto parseArgElem = [&]() -> ParseResult {
        OpAsmParser::Argument argument;
        auto arg_present = parser.parseOptionalArgument(argument, true, false);
        if (arg_present.has_value()) {
            if (failed(arg_present.value())) {
                return failure(); // Present but malformed.
            }

            // Reject this if the preceding argument was missing a name.
            if (!entry_args.empty() && entry_args.back().ssaName.name.empty()) {
                return parser.emitError(argument.ssaName.location, "expected type instead of SSA identifier");
            }
        } else {
            argument.ssaName.location = parser.getCurrentLocation();
            // Oterwise we just have a type list without SSA names. Reject this if the preceding argument had a name.
            if (!entry_args.empty() && !entry_args.back().ssaName.name.empty()) {
                return parser.emitError(argument.ssaName.location, "expected SSA identifier");
            }

            if (parser.parseType(argument.type) || parser.parseOptionalLocationSpecifier(argument.sourceLoc)) {
                return failure();
            }
        }
        entry_args.push_back(argument);
        return success();
    };
    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseArgElem)) {
        return failure();
    }

    // Parse function signature: Result type
    bool has_result_type = succeeded(parser.parseOptionalArrow());
    SmallVector<Type> result_types;
    if (has_result_type) {
        Type result_type;
        if (parser.parseType(result_type)) {
            return failure();
        }
        result_types.push_back(result_type);
    }

    // Add function signature
    SmallVector<Type> arg_types;
    arg_types.reserve(entry_args.size());
    for (auto& arg : entry_args) {
        arg_types.push_back(arg.type);
    }
    Type function_type = builder.getFunctionType(arg_types, result_types);
    if (!function_type) {
        return parser.emitError(signature_loc, "failed to construct function type");
    }
    result.addAttribute("function_type", TypeAttr::get(function_type));

    // Parse optional function body
    auto* body = result.addRegion();
    SMLoc body_loc = parser.getCurrentLocation();
    OptionalParseResult parse_result = parser.parseOptionalRegion(*body, entry_args, false);
    if (parse_result.has_value()) {
        if (failed(*parse_result)) {
            return failure();
        }
        // Function body was parsed, make sure ist not empty.
        if (body->empty()) {
            return parser.emitError(body_loc, "expected non-empty function body");
        }
    }

    return success();
}

LogicalResult ReturnOp::verify() {
    auto sdfg_op = cast<SDFGOp>(this->getParentOp());
    const auto& results = sdfg_op.getFunctionType().getResults();
    auto operand = this->getOperand();
    if (operand) {
        if (results.size() == 0) {
            return this->emitOpError("has operand, but enclosing SDFG (@")
                   << sdfg_op.getSymName() << ") has no result type";
        }
        assert(results.size() == 1);
        if (results.front() != operand.getType()) {
            return this->emitError() << "type of return operand (" << operand.getType()
                                     << ") doesn't match SDFG result type (" << results.front() << ") in SDFG @"
                                     << sdfg_op.getSymName();
        }
    } else {
        if (results.size() == 0) {
            return success();
        } else if (results.size() == 1) {
            return this->emitOpError("has no operand, but enclosing SDFG (@")
                   << sdfg_op.getSymName() << ") has a result type";
        }
    }
    return success();
}

} // namespace sdfg
} // namespace mlir
