#include "toy_dialect.h"
#include "toy_dialect.cpp.inc"

using namespace mlir;
using namespace mlir::toy;

// Register the dialect with the MLIR context
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy_ops.cpp.inc"
    >();
}

#define GET_OP_CLASSES
#include "toy_ops.cpp.inc"
