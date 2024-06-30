#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "toy/toy_dialect.h"
#include "traverse.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::toy::registerToyDialect(registry);

  mlir::registerPrintNestingPass();

  // Delegate to the MLIR utility for parsing and pass management.
  return mlir::MlirOptMain(argc, argv, "traverse", registry)
                          .succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
}