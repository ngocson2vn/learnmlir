#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "Interface.h"

// Include generated Dialect declarations and definitions
#include "ExampleDialect.h.inc"

#define GET_OP_CLASSES
#include "ExampleOps.h.inc"
