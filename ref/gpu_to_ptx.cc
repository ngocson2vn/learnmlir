/*
clang++ -std=c++17 gpu_to_ptx.cc -o gpu_to_ptx \
  `llvm-config --cxxflags --ldflags --libs` \
  -I/path/to/mlir/include -L/path/to/mlir/lib -lMLIR

Example Input (input.mlir)
```mlir
module {
  func.func @my_kernel(%arg0: memref<?xf32>) {
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
               threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
      %val = memref.load %arg0[%tx] : memref<?xf32>
      gpu.terminator
    }
    return
  }
}

./gpu_to_ptx input.mlir -o output.ll

llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70 output.ll -o output.ptx
```
*/

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

using namespace mlir;

int main(int argc, char **argv) {
  // Command-line options for input/output files
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input MLIR file>"),
      llvm::cl::init("-"), llvm::cl::value_desc("filename"));
  
  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output LLVM IR file"),
      llvm::cl::init("output.ll"), llvm::cl::value_desc("filename"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "GPU to PTX Lowering Pipeline\n");

  // Initialize MLIR context and register dialects
  MLIRContext context;
  context.getOrLoadDialect<gpu::GPUDialect>();
  context.getOrLoadDialect<NVVM::NVVMDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();
  context.getOrLoadDialect<arith::ArithmeticDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<StandardOpsDialect>();

  // Load the input MLIR file
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return 1;
  }

  // Parse the input MLIR
  auto owningModule = parseSourceFile<ModuleOp>(fileOrErr->get()->getMemBufferRef(), &context);
  if (!owningModule) {
    llvm::errs() << "Error: Failed to parse MLIR input file\n";
    return 1;
  }

  // Create a pass manager
  PassManager pm(&context);
  
  // Build the lowering pipeline
  pm.addPass(createConvertGPUToNVVMPass());           // GPU -> NVVM
  pm.addPass(createConvertSCFToCFPass());             // SCF -> Control Flow
  pm.addPass(createConvertArithmeticToLLVMPass());    // Arithmetic -> LLVM
  pm.addPass(createConvertNVVMToLLVMPass());          // NVVM -> LLVM
  pm.addPass(createLowerToLLVMPass());                // LLVM Dialect -> LLVM IR

  // Optional: Add optimization passes
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Run the pipeline
  if (failed(pm.run(*owningModule))) {
    llvm::errs() << "Error: Pass pipeline failed\n";
    return 1;
  }

  // Export to LLVM IR
  llvm::raw_fd_ostream output(outputFilename, std::error_code());
  if (std::error_code ec = output.error()) {
    llvm::errs() << "Could not open output file: " << ec.message() << "\n";
    return 1;
  }

  auto llvmModule = translateModuleToLLVMIR(*owningModule, context);
  if (!llvmModule) {
    llvm::errs() << "Error: Failed to translate to LLVM IR\n";
    return 1;
  }
  llvmModule->print(output, nullptr);
  output.close();

  llvm::outs() << "LLVM IR written to " << outputFilename << "\n";
  llvm::outs() << "To generate PTX, run:\n";
  llvm::outs() << "  llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70 " 
               << outputFilename << " -o output.ptx\n";

  return 0;
}