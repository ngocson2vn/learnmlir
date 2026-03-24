#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

// Include generated Dialect declarations and definitions
#include "ExampleDialect.h.inc"

// Attributes
#define GET_ATTRDEF_CLASSES
#include "ExampleAttributes.h.inc"

#define GET_OP_CLASSES
#include "ExampleOps.h.inc"
