// MLIR IRs
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

// MLIR Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/OptimizeForNVVM.h"

// MLIR Passes
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// Toy Compiler headers
#include "frontend/toy_dialect.h"
#include "passes.h"
#include "middleend.h"

using namespace mlir;

namespace toy {
namespace compiler {
namespace middleend {

LogicalResult lower(mlir::ModuleOp& module) {
  auto& context = *module.getContext();

  // Load necessary dialects
  context.getOrLoadDialect<mlir::toy::ToyDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();

  // Set up the pass manager
  PassManager pm(&context);

  // Toy to Std
  pm.addPass(mlir::toy::createConvertToyToStdPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // To MemRef
  pm.addPass(mlir::toy::createConvertTensorToMemRefPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // To Loops
  pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Apply the pass
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass execution failed\n";
    return failure();
  }

  return success();
}

} // namespace middleend
} // namespace compiler
} // namespace toy
