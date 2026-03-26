#include "Dialect.h"

#include "ExampleDialect.cpp.inc"

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
}

} // namespace example
} // namespace mlir