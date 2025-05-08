#include "toy_dialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::toy;

// Register the dialect with the MLIR context
ToyDialect::ToyDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<ToyDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "toy_ops.cpp.inc"
      >();
}

// Include the auto-generated operation definitions
#define GET_OP_CLASSES
#include "toy_ops.cpp.inc"