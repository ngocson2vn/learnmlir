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

  // // Load an MLIR module with GPU dialect (example input)
  // OwningOpRef<ModuleOp> module;
  // {
  //   auto file = openInputFile(inputFilename); // Assume input.mlir contains gpu.alloc
  //   if (!file) {
  //     llvm::errs() << "Failed to open input file " << inputFilename << "\n";
  //     return 1;
  //   }

  //   // Parse the input mlir.
  //   llvm::SourceMgr sourceMgr;
  //   sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  //   mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  //   if (!module) {
  //     llvm::errs() << "Error can't load file " << inputFilename << "\n";
  //     return 1;
  //   }
  // }

  OpBuilder builder(&context);
  auto loc = UnknownLoc::get(&context);
  ModuleOp module = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());

  auto indexType = builder.getIndexType();
  auto kernelFunc = builder.create<func::FuncOp>(loc, "kernel", builder.getFunctionType({indexType}, {}));
  Block* kernelBody = kernelFunc.addEntryBlock();
  builder.setInsertionPointToStart(kernelBody);

  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  // Value c1 = builder.create<arith::ConstantFloatOp>(loc, llvm::APFloat(1.0f), builder.getF32Type());
  // MemRefType xType = MemRefType::get({}, builder.getF32Type(), MemRefLayoutAttrInterface{});
  // Value x = builder.create<memref::AllocaOp>(loc, xType);

  // auto memset = builder.create<gpu::MemsetOp>(loc, TypeRange{builder.getType<mlir::gpu::AsyncTokenType>()}, ValueRange{c1}, x, c1);

  // auto launchOp = builder.create<gpu::LaunchOp>(
  //   loc, 
  //   c1, c1, c1,               // Grid size
  //   c1, c1, c1                // Block size
  // );

  // builder.setInsertionPointToStart(&launchOp.getBody().front());

  // Dummy or prior async token (e.g., from a gpu::LaunchOp)
  auto dummyLaunchOp = builder.create<gpu::LaunchOp>(loc, 
    c1, c1, c1,
    c1, c1, c1,
    nullptr,
    builder.getType<mlir::gpu::AsyncTokenType>(), ValueRange{}
  );

  auto a = dummyLaunchOp.getAsyncToken();
  auto memrefType = MemRefType::get({ShapedType::kDynamicSize}, builder.getF32Type(), MemRefLayoutAttrInterface{}, builder.getI32IntegerAttr(3));
  builder.create<gpu::AllocOp>(loc, memrefType, a.getType(), ValueRange{a}, ValueRange{kernelBody->getArgument(0)}, ValueRange{});

  // builder.setInsertionPointToEnd(&launchOp.getBody().front());
  // builder.create<gpu::TerminatorOp>(loc);
  
  builder.create<func::ReturnOp>(loc);


  auto input = openOutputFile("lower_gpu_ops/input.mlir");
  if (!input) {
    std::cerr << "Failed to open input file" << std::endl;
    return 1;
  }
  input->keep();
  module->print(input->os());
  input->os().flush();

  // Set up the pass manager
  PassManager pm(&context);
  
  // Option 1: Lower GPU to NVVM
  pm.addPass(createGpuKernelToNvvmPass());
  pm.addPass(createGpuToLLVMConversionPass());
  
  // Option 2: Lower GPU to LLVM (comment out NVVM pass if using this)
  // pm.addPass(createConvertGPUToLLVMPass());

  // Option 3: Chain NVVM to LLVM (uncomment both if needed)
  // pm.addPass(createConvertNVVMToLLVMPass());

  // Run the passes
  if (failed(pm.run(module))) {
    std::cerr << "Pass pipeline failed" << std::endl;
    return 1;
  }

  // Print the result
  module->dump();

  // Optionally save to file
  auto output = openOutputFile("output.mlir");
  if (!output) {
    std::cerr << "Failed to open output file" << std::endl;
    return 1;
  }
  module->print(output->os());
  output->keep();

  return 0;
}
