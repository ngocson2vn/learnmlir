#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/Support/Debug.h"

#include "Dialect.h"


template<typename T>
void printType() {
  std::string type = __PRETTY_FUNCTION__;
  llvm::outs() << type << "\n";
}

using namespace mlir;

int main() {
  // Setup MLIR Context and Builder
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<example::ExampleDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  // Create our operations
  auto defaultOp = builder.create<example::DefaultOp>(loc);
  auto customOp = builder.create<example::CustomOp>(loc);

  // Generic lambda that accepts ANY operation and attempts to cast it
  // to our ExampleOpInterface.
  auto printMagicNumber = [](Operation *op) {
    // Interface cast (concept-based polymorphism)
    if (auto iface = mlir::dyn_cast<example::ExampleOpInterface>(op)) {
      llvm::outs() << op->getName() << " magic number: " 
                   << iface.getMagicNumber() << "\n";
    } else {
      llvm::outs() << op->getName() << " does not implement ExampleOpInterface.\n";
    }
  };

  // Test the interface hooks
  printMagicNumber(customOp);
  printMagicNumber(defaultOp);

  auto lhs = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 23, 32);
  auto rhs = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 67, 32);
  auto addOp = builder.create<arith::AddIOp>(builder.getUnknownLoc(), lhs, rhs);

  auto iface = llvm::cast<example::ExampleOpInterface>(addOp.getOperation());
  llvm::outs() << addOp->getName() << " magic number: " << iface.getMagicNumber() << "\n";

  return 0;
}
