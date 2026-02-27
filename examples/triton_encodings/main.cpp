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

#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"),
    llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Triton Encodings\n");

  // Set up the MLIR context
  MLIRContext context;
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  registerAllDialects(context);

  // Create a ModuleOp
  OpBuilder builder(&context);
  auto loc = UnknownLoc::get(&context);
  ModuleOp module = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());

  auto indexType = builder.getIndexType();
  auto kernelFunc = builder.create<func::FuncOp>(loc, "main", builder.getFunctionType({indexType}, {}));
  Block* kernelBody = kernelFunc.addEntryBlock();
  builder.setInsertionPointToStart(kernelBody);

  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<func::ReturnOp>(loc, c1);

  // Print the resulting module
  llvm::outs() << "\nMLIR module:\n";
  module->print(llvm::outs());
  llvm::outs() << "\nAll done\n";
  return 0;
}
