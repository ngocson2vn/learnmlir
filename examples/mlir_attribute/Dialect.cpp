#include <tuple>

#include "Dialect.h"

#include "ExampleDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ExampleAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "ExampleOps.cpp.inc"

namespace mlir {
namespace example {

// Initialize the dialect and register its operations
void ExampleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ExampleOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "ExampleAttributes.cpp.inc"
      >();
}

/*
===============================================
  BlockedEncodingAttr printer and parser
===============================================
*/

// 1. Manually implement the printer
void BlockedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<" << getNumBlocks() << ">";
}

// 2. Manually implement the parser
Attribute BlockedEncodingAttr::parse(AsmParser &parser, Type type) {
  int numBlocks = 0;
  
  // Parse the "<"
  if (parser.parseLess())
    return Attribute();

  // Parse the integer numBlocks
  if (parser.parseInteger(numBlocks))
    return Attribute();

  // Parse the ">"
  if (parser.parseGreater())
    return Attribute();

  // Create and return the attribute instance
  return BlockedEncodingAttr::get(parser.getContext(), numBlocks);
}

::llvm::LogicalResult BlockedEncodingAttr::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int numBlocks) {
  if (numBlocks < 0) {
    emitError() << "numBlocks is negative\n";
    return failure();
  }

  return success();
}

/*
===============================================
  BlockedEncodingAttr printer and parser
===============================================
*/

// 1. Manually implement the printer
void DistributedShardingAttr::print(AsmPrinter &printer) const {
  printer << "<{";

  // devices
  printer << "devices = [";
  auto devices = getDevices();
  for (int i = 0; i < devices.size() - 1; i++) {
    printer << devices[i] << ", ";
  }
  printer << devices.back() << "]";

  // splitAxis
  auto splitAxis = getSplitAxis();
  printer << ", split_axis = " << splitAxis;
  printer << "}>";
}

// 2. Manually implement the parser
Attribute DistributedShardingAttr::parse(AsmParser &parser, Type type) {
  // int devices = ;
  
  // // Parse the "<"
  // if (parser.parseLess())
  //   return Attribute();

  // // Parse the integer numBlocks
  // if (parser.parseInteger(numBlocks))
  //   return Attribute();

  // // Parse the ">"
  // if (parser.parseGreater())
  //   return Attribute();

  // Create and return the attribute instance
  // return BlockedEncodingAttr::get(parser.getContext(), numBlocks);
  return DistributedShardingAttr::get(parser.getContext(), {0, 1, 2, 3}, 0);
}

::llvm::LogicalResult DistributedShardingAttr::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, std::vector<int> devices, int splitAxis) {
  if (devices.empty()) {
    emitError() << "DistributedShardingAttr's devices is empty\n";
    return failure();
  }

  if (splitAxis < 0) {
    emitError() << "DistributedShardingAttr's splitAxis is negative\n";
    return failure();
  }

  llvm::outs() << "DistributedShardingAttr's parameters are valid!\n";
  return success();
}

} // namespace example
} // namespace mlir