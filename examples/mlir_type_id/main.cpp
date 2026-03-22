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
  mlir::registerAllDialects(context);

  auto typeID1 = mlir::TypeID::get<mlir::IntegerType>();
  llvm::outs() << "typeID1: " << typeID1.getAsOpaquePointer() << "\n";

  auto typeID2 = mlir::IntegerType::getTypeID();
  llvm::outs() << "typeID2: " << typeID2.getAsOpaquePointer() << "\n";

  return 0;
}
