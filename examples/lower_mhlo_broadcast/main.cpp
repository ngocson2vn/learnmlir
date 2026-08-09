#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Debug.h"

// MLIR Dialects
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

// MLIR passes
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

// MHLO
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/IR/register.h"
#include "transforms/passes.h"


template<typename T>
void printType() {
  std::string type = __PRETTY_FUNCTION__;
  llvm::outs() << type << "\n";
}

using namespace mlir;

int main(int argc, char** argv) {
  //============================================================
  // 
  //============================================================
  static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<module.mlir>"),
    llvm::cl::init("-"),
    llvm::cl::value_desc("filename")
  );

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "examples/lower_mhlo_broadcast/main.cpp\n");

  //============================================================
  // Create MLIR context and load dialects
  //============================================================
  DialectRegistry registry;
  registry.insert<func::FuncDialect>();
  registry.insert<memref::MemRefDialect>();
  registerAllDialects(registry);
  mhlo::registerAllMhloDialects(registry);

  // Initialize MLIR context
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Double-check loaded dialects
  // for (auto name : context.getAvailableDialects()) {
  //   llvm::outs() << "Loaded " << name << "\n";
  // }

  //============================================================
  // Load module.mlir
  //============================================================
  OwningOpRef<ModuleOp> moduleRef;
  {
    llvm::outs() << "Load " << inputFilename << "\n";
    auto file = openInputFile(inputFilename); // Assume input.mlir contains gpu.alloc
    if (!file) {
      llvm::errs() << "Failed to open input file " << inputFilename << "\n";
      return 1;
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    moduleRef = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!moduleRef) {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      return 1;
    }
  }

  auto module = moduleRef.get();
  module.dump();

  //============================================================
  // Lower module
  //============================================================
  context.disableMultithreading();
  mlir::PassManager pm(module.getContext());
  std::string errorMessage;
  std::string outputFile = "./lowering.mlir";
  auto output = mlir::openOutputFile(outputFile, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    std::terminate();
  }
  output->keep();

  mlir::OpPrintingFlags flag{};
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
    output->os(), flag
  );

  pm.addPass(mlir::createComputeOpAndFuncBufferizePass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(mlir::createFinalBufferizePass());
  pm.addPass(mlir::createCanonicalizerPass());

  llvm::outs() << "\nStart lowering module\n";
  if (failed(pm.run(module))) {
    llvm::errs() << "Failed to lower module!\n";
    return 1;
  }

  llvm::outs() << "Finished lowering module\n";
  llvm::outs() << "Please check " << outputFile << "\n";

  return 0;
}
