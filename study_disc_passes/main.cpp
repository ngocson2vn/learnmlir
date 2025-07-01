#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

#include "stablehlo/dialect/ChloOps.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "lhlo/transforms/passes.h"
#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/passes.h"
#include "lhlo_gpu/IR/lhlo_gpu_ops.h"

// DISC IRs
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/IR/disc_ral_ops.h"

// DISC passes
#include "mlir/disc/transforms/disc_passes.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"),
    llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Toy dialect lowering demo\n");

  // Set up the MLIR context
  mlir::registerAllPasses();
  mlir::registerTensorFlowPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::disc_ral::registerAllDiscPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);
  // mlir::registerLLVMDialectTranslation(registry);
  // mlir::registerNVVMDialectTranslation(registry);
  registry.insert<mlir::mhlo::MhloDialect>();
  // registry.insert<mlir::mhlo_disc::MhloDiscDialect>();
  registry.insert<mlir::chlo::ChloDialect>();
  registry.insert<mlir::lmhlo::LmhloDialect>();
  // registry.insert<mlir::lmhlo_disc::LmhloDiscDialect>();
  // registry.insert<mlir::lmhlo_gpu::LmhloGpuDialect>();
  registry.insert<mlir::disc_shape::DISCShapeDialect>();
  registry.insert<mlir::disc_ral::RalDialect>();
  registry.insert<mlir::TF::TensorFlowDialect>();
  registry.insert<mlir::torch::Torch::TorchDialect>();

  MLIRContext context(registry);

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

  // Set up the pass manager
  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(disc_ral::createDiscLhloLegalizeRootsToParallelLoopsPass());

  // Apply the pass
  if (failed(pm.run(module.get()))) {
    llvm::errs() << "Pass execution failed\n";
    return 1;
  }

  // Print the resulting module
  llvm::outs() << "Lowered MLIR:\n";
  module->print(llvm::outs());
  return 0;
}
