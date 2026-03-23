#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

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
  context.getOrLoadDialect<example::ExampleDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  // Create our operations
  auto defaultOp = builder.create<example::DefaultOp>(loc);
  auto customOp = builder.create<example::CustomOp>(loc);

  // Generic lambda that accepts ANY operation and attempts to cast it
  // to our ExampleOpInterface.
  auto printMagicNumber = [](Operation *op) {
    // Interface cast (concept-based polymorphism)
    if (auto iface = dyn_cast<example::ExampleOpInterface>(op)) {
      llvm::outs() << op->getName() << " magic number: " 
                   << iface.getMagicNumber() << "\n";
    } else {
      llvm::outs() << op->getName() << " does not implement ExampleOpInterface.\n";
    }
  };

  // Test the interface hooks
  // printMagicNumber(defaultOp);
  printMagicNumber(customOp);

  return 0;
}
