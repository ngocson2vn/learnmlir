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

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"),
    llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Triton Encodings\n");

  // Set up the MLIR context
  MLIRContext context;
  registerAllDialects(context);
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<mlir::triton::TritonDialect>();
  context.loadDialect<mlir::triton::gpu::TritonGPUDialect>();

  // Create a ModuleOp
  OpBuilder builder(&context);
  auto loc = UnknownLoc::get(&context);
  ModuleOp module = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(module.getBody());

  // Create a RankedTensorType instance
  llvm::SmallVector<int64_t, 2> shape{64, 32};
  mlir::Type elementType = builder.getF32Type();

  // 1. Create the specific encoding attribute
  llvm::SmallVector<unsigned, 2> sizePerThread{2, 2};
  llvm::SmallVector<unsigned, 2> order{0, 1};
  unsigned numWarps = 2;
  unsigned numThreadsPerWarp = 32;
  auto CTALayout = triton::gpu::CTAEncodingAttr::getDefault(&context, 2);
  mlir::Attribute encoding = triton::gpu::BlockedEncodingAttr::get(&context, shape, sizePerThread, order, numWarps, numThreadsPerWarp, CTALayout);

  // 2. Build the RankedTensorType with the encoding attached
  auto tensorType = mlir::RankedTensorType::get(shape, elementType, encoding);
  // llvm::outs() << "tensorType: " << tensorType << "\n";

  // Create FuncOp
  auto indexType = builder.getIndexType();
  auto kernelFunc = builder.create<func::FuncOp>(loc, "main", builder.getFunctionType({tensorType}, {indexType}));
  Block* kernelBody = kernelFunc.addEntryBlock();
  builder.setInsertionPointToStart(kernelBody);

  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<func::ReturnOp>(loc, c1);

  // Print the resulting module
  llvm::outs() << "MLIR module:\n";
  module->print(llvm::outs());
  llvm::outs() << "\nAll done\n";
  return 0;
}
