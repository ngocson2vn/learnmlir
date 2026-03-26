#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
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
  defaultOp->setAttr("magic_number", IntegerAttr::get(builder.getI64Type(), 55));

  auto customOp = builder.create<example::CustomOp>(loc);
  customOp->setAttr("magic_number", IntegerAttr::get(builder.getI64Type(), 100));

  // // Explicitly run the verifier!
  // // This will trigger calling VerifyMagicNumberTrait::verifyTrait()
  // if (::mlir::failed(::mlir::verify(customOp))) {
  //     llvm::errs() << "CustomOp verification failed!\n";
  //     return 1;
  // }

  llvm::outs() << defaultOp << "\n";
  llvm::outs() << defaultOp->getName() << " magic number = " << defaultOp.getMagicNumber() << "\n";
  llvm::outs() << defaultOp->getName() << " magic number even? " << (defaultOp.isMagicNumberEven() ? "true" : "false") << "\n";

  llvm::outs() << "\n";

  llvm::outs() << customOp << "\n";
  llvm::outs() << customOp->getName() << " magic number = " << customOp.getMagicNumber() << "\n";
  llvm::outs() << customOp->getName() << " magic number even? " << (customOp.isMagicNumberEven() ? "true" : "false") << "\n";

  return 0;
}
