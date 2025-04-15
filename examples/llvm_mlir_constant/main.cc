#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Transforms/gpu_passes.h"

#include <iostream>

using namespace mlir;

int main(int argc, char** argv) {
  // Initialize MLIR context and register dialects
  MLIRContext context;

  static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input toy file>"),
    llvm::cl::init("-"),
    llvm::cl::value_desc("filename")
  );

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "lower_gpu_ops\n");

  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<gpu::GPUDialect>();
  context.getOrLoadDialect<NVVM::NVVMDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  registerAllDialects(context);


  OpBuilder builder(&context);
  auto loc = UnknownLoc::get(&context);
  ModuleOp module = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());

  // auto indexType = builder.getIndexType();
  auto i32Type = IntegerType::get(&context, 32);
  auto funcType = LLVM::LLVMFunctionType::get(&context, i32Type, {i32Type}, false);
  auto mainFunc = builder.create<LLVM::LLVMFuncOp>(loc, "main", funcType);
  Block* mainBody = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(mainBody);
  Value c0 = builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), 0);
  Value c32 = builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), 32);
  builder.create<LLVM::ReturnOp>(loc, ValueRange{c0});

  module->dump();

  // Lower to LLVM module.
  mlir::registerLLVMDialectTranslation(*module->getContext());
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Could not translate MLIR module to LLVM IR" << "\n";
    return 1;
  }
  llvmModule->setModuleIdentifier("sony");


  // Print the result
  llvmModule->dump();

  // Optionally save to file
  auto output = openOutputFile("llvm_mlir_constant/output.mlir");
  if (!output) {
    std::cerr << "Failed to open output file" << std::endl;
    return 1;
  }
  output->keep();
  llvmModule->print(output->os(), nullptr);

  return 0;
}
