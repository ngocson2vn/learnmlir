#pragma once

#include <vector>
#include "llvm/ADT/Hashing.h"

// Hash functor for std::vector<int>
// Template Two-phase Name Lookup:
// Phase 1: At the point of Definition, the compiler searchs for all names from above to this point
// Phase 2: At the point of Instantiation, the compiler will perform ADL to find more candidates of `hash_value`
// Therefore, namespace must be `std` to make ADL (Argument-dependent Lookup) work
namespace std {

template <typename T> llvm::hash_code hash_value(const std::vector<T>& vec) {
  return llvm::hash_combine_range(vec.begin(), vec.end());
}

} // namespace std

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
