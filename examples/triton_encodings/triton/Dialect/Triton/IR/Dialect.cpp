#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;

#include "triton/Dialect/Triton/IR/Dialect.cpp.inc"

void TritonDialect::initialize() {
  registerTypes();

//   addOperations<
// #define GET_OP_LIST
// #include "triton/Dialect/Triton/IR/Ops.cpp.inc"
//       >();

  // We can also add interface here.
  // addInterfaces<TritonInlinerInterface>();
}

Operation *TritonDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}