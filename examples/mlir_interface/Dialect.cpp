#include "Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "ExampleDialect.cpp.inc"

#define GET_OP_CLASSES
#include "ExampleOps.cpp.inc"

namespace {

class ArithAddIOpExampleModel : public mlir::example::ExampleOpInterface::ExternalModel<ArithAddIOpExampleModel, mlir::arith::AddIOp> {
};

}

namespace mlir {
namespace example {

// Initialize the dialect and register its operations
void ExampleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ExampleOps.cpp.inc"
      >();

  ::mlir::arith::AddIOp::attachInterface<ArithAddIOpExampleModel>(*getContext());
}

// Implement the overridden interface method for CustomOp
int CustomOp::getMagicNumber() {
  return 99; // Overrides the default 42
}

} // namespace example
} // namespace mlir