#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Debug.h"


void printType(mlir::Type type) {
  llvm::outs() << "type: " << type << "\n";
}

int main(int argc, char **argv) {
  // Set up the MLIR context
  mlir::MLIRContext context;
  // mlir::registerAllDialects(context);

  // Create an IntegerType
  auto i32Type = mlir::IntegerType::get(&context, 32);
  printType(i32Type);

  return 0;
}
