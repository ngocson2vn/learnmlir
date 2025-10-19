#include "mlir/InitAllExtensions.h"

// MLIR IRs
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

// MLIR Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/OptimizeForNVVM.h"

// MLIR Passes
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

#include "passes.h"
#include "backend.h"
#include "llvm_utils.h"
#include "cuda_utils.h"
#include "nvidia_backend.h"

using namespace mlir;

namespace toy {
namespace compiler {
namespace backend {

llvm::LogicalResult lower(mlir::ModuleOp& module, int capability) {
  auto& context = *module.getContext();

  // Load necessary dialects

  // Set up the pass manager
  context.disableMultithreading();
  PassManager pm(&context);
  std::string errorMessage;

  // lowering
  auto output = mlir::openOutputFile("lowering.mlir", &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  mlir::OpPrintingFlags printFlag{};
  pm.enableIRPrinting(
    /*shouldPrintBeforePass=*/[](mlir::Pass* p, mlir::Operation* op) {
      return false;
    },
    /*shouldPrintAfterPass=*/[](mlir::Pass* p, mlir::Operation * op) {
      return true;
    },
    /*printModuleScope=*/true, 
    /*printAfterOnlyOnChange=*/true,
    /*printAfterOnlyOnFailure=*/false, 
    output->os(), printFlag
  );
  output->keep();

  //============================================================================
  // To GPU
  //============================================================================
  mlir::SmallVector<int64_t> tileSizes{128};
  pm.addNestedPass<func::FuncOp>(mlir::toy::createTileLoopsPass(tileSizes));

  pm.addNestedPass<func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
  pm.addPass(mlir::createConvertParallelLoopToGpuPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::createGpuLaunchSinkIndexComputationsPass());

  const std::string dataLayoutStr = "#dlti.dl_spec<#dlti.dl_entry<index, 64 : i64>>";
  mlir::GpuKernelOutliningPassOptions outliningOptions{dataLayoutStr};
  pm.addPass(mlir::createGpuKernelOutliningPass(outliningOptions));
  
  pm.addPass(mlir::createLowerAffinePass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  //============================================================================
  // Lower GPUModuleOp to CUBIN
  //============================================================================
  DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);

  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  auto& kernelPm = pm.nest<::mlir::gpu::GPUModuleOp>();
  kernelPm.addPass(mlir::createConvertGpuOpsToNVVMOps());
  kernelPm.addPass(mlir::createConvertNVVMToLLVMPass());
  kernelPm.addPass(mlir::LLVM::createNVVMOptimizeForTargetPass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::toy::createGpuModuleToCubinPass());

  // Apply the pass
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass execution failed\n";
    return failure();
  }

  //============================================================================
  // Lower host code
  //============================================================================
  pm.clear();
  pm.addPass(mlir::createConvertToLLVMPass());
  pm.addPass(mlir::createGpuToLLVMConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass execution failed\n";
    return failure();
  }

  llvm::outs() << "\nFinal MLIR module:\n";
  module.dump();
  llvm::outs() << "\n";

  llvm::LLVMContext llvmContext;
  auto llvmMod = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmMod) {
    llvm::errs() << "Failed to translate module to LLVM IR\n";
    return failure();
  }



  return success();
}

} // namespace middleend
} // namespace compiler
} // namespace toy
