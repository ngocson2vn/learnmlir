#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"

#include <iostream>

using namespace mlir;

int main(int argc, char** argv) {
  // Register any command line options.

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::lmhlo::LmhloDialect>();
  mlir::registerAllDialects(registry);

  // Initialize MLIR context and register dialects
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OpBuilder builder(&context);
  auto loc = UnknownLoc::get(&context);
  ModuleOp module = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());

  auto i32MemRefType = mlir::MemRefType::get({}, builder.getI32Type());
  auto mainFunc = builder.create<func::FuncOp>(loc, "main", builder.getFunctionType({i32MemRefType}, {}));
  Block* mainBody = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(mainBody);

  // Common type
  auto memrefType = mlir::MemRefType::get({}, i32MemRefType);

  // Create a constant attribute with value 42
  auto denseAttr = mlir::DenseElementsAttr::get(i32MemRefType, builder.getI32IntegerAttr(21));
  auto c21Buffer = builder.create<mlir::memref::AllocOp>(loc, memrefType);
  builder.create<mlir::lmhlo::ConstantOp>(loc, denseAttr, c21Buffer);

  // Create AddOp
  auto outBuffer = builder.create<mlir::memref::AllocOp>(loc, memrefType);
  auto addOp = builder.create<lmhlo::AddOp>(loc, c21Buffer, c21Buffer, outBuffer, nullptr);
  builder.create<func::ReturnOp>(loc, ValueRange{outBuffer});

  auto op = ::llvm::dyn_cast<lmhlo::LmhloOp>(addOp.getOperation());
  if (!!op) {
    llvm::outs() << "\nop: " << op << "\n\n";
  }

  // mainFunc.walk([&](lmhlo::LmhloOp op) {
  //   llvm::outs() << "op: " << op << "\n\n";
  // });

  // Set up the pass manager
  // PassManager pm(&context);
  
  // Option 1: Lower GPU to NVVM
  // pm.addPass(createGpuKernelToNvvmPass());
  // pm.addPass(createGpuToLLVMConversionPass());
  
  // Option 2: Lower GPU to LLVM (comment out NVVM pass if using this)
  // pm.addPass(createConvertGPUToLLVMPass());

  // Option 3: Chain NVVM to LLVM (uncomment both if needed)
  // pm.addPass(createConvertNVVMToLLVMPass());

  // Run the passes
  // if (failed(pm.run(module))) {
  //   std::cerr << "Pass pipeline failed" << std::endl;
  //   return 1;
  // }


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
