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

// 1. Manually implement the printer
void BlockedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<" << getRank() << ">";
}

// 2. Manually implement the parser
Attribute BlockedEncodingAttr::parse(AsmParser &parser, Type type) {
  int rank = 0;
  
  // Parse the "<"
  if (parser.parseLess())
    return Attribute();

  // Parse the integer rank
  if (parser.parseInteger(rank))
    return Attribute();

  // Parse the ">"
  if (parser.parseGreater())
    return Attribute();

  // Create and return the attribute instance
  return BlockedEncodingAttr::get(parser.getContext(), rank);
}

} // namespace example
} // namespace mlir