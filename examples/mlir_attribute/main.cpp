#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Debug.h"

#include "Dialect.h"


template<typename T>
void printType() {
  std::string type = __PRETTY_FUNCTION__;
  llvm::outs() << type << "\n";
}

using namespace mlir;

int main() {
// Setup MLIR Context
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::example::ExampleDialect>();
  mlir::OpBuilder builder(&context);

  // 1. Instantiate our custom attribute (e.g., specifying an alignment of 128)
  // ODS automatically generates the static `get()` method for us based on the parameters.
  auto encodingAttr1 = mlir::example::BlockedEncodingAttr::get(&context, /*numBlocks=*/4);
  auto encodingAttr2 = mlir::example::DistributedShardingAttr::get(&context, {0, 1, 2, 3}, 0);
  // auto invalidEncodingAttr = mlir::example::DistributedShardingAttr::get(&context, {0, 1, 2, 3}, -1);

  // 2. Define the shape and element type of the tensor
  llvm::SmallVector<int64_t> shape = {10, 20};
  mlir::Type elementType = builder.getF32Type();

  // 3. Create the BlockedTensorType, passing our attribute as the 3rd argument (encoding)
  auto tensorType1 = mlir::RankedTensorType::get(shape, elementType, encodingAttr1);
  auto tensorType2 = mlir::RankedTensorType::get(shape, elementType, encodingAttr2);

  // 4. Print the resulting type to the console
  llvm::outs() << "Generated Tensor Type 1: " << tensorType1 << "\n";
  llvm::outs() << "Generated Tensor Type 2: " << tensorType2 << "\n";

  return 0;
}
