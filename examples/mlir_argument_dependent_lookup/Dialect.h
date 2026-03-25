#pragma once

#include <vector>
#include "llvm/ADT/Hashing.h"

// Hash functor for `std::vector<int>`
// NOTE: Though `hash_value` is defined here but the argument is `std::vector<int>`, 
// so ADL searches `namespace std`. Because `hash_value` in `namespace llvm`. So, ADL fails.
// To make ADL work, we need to change namespace from `llvm` to `std`.
namespace llvm {

template <typename T> hash_code hash_value(const std::vector<T>& vec) {
  return hash_combine_range(vec.begin(), vec.end());
}

} // namespace llvm

//========================================================
// Uncomment the following code to make ADL work
//========================================================
// namespace std {

// template <typename T> llvm::hash_code hash_value(const std::vector<T>& vec) {
//   return llvm::hash_combine_range(vec.begin(), vec.end());
// }

// } // namespace std

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
