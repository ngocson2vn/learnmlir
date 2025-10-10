#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "toy_dialect.h"
#include "toy_passes.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"),
    llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Toy dialect lowering demo\n");

  // Set up the MLIR context
  MLIRContext context;
  context.getOrLoadDialect<toy::ToyDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<bufferization::BufferizationDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();

  // Load the input MLIR file
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return 1;
  }

  // Parse the input MLIR
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error parsing input file\n";
    return 1;
  }

  llvm::outs() << "Before lowering:\n";
  module->print(llvm::outs());
  llvm::outs() << "\n\n";

  // Set up the pass manager
  PassManager pm(&context);
  pm.addPass(mlir::toy::createConvertToyToArithPass());
  pm.addPass(mlir::toy::createConvertTensorToMemRefPass());
  // pm.addPass(mlir::createCanonicalizerPass());

  // Apply the pass
  if (failed(pm.run(module.get()))) {
    llvm::errs() << "Pass execution failed\n";
    return 1;
  }

  // Print the resulting module
  llvm::outs() << "\nLowered MLIR:\n";
  module->print(llvm::outs());
  llvm::outs() << "\nAll done\n";
  return 0;
}
