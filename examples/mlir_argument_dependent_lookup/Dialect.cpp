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
  int splitAxis = getSplitAxis();
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

} // namespace example
} // namespace mlir