#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/TypeID.h"

#include "toy_dialect.h.inc"

#define GET_OP_CLASSES
#include "toy_ops.h.inc"