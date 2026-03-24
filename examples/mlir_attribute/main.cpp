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
  auto encodingAttr = mlir::example::BlockedEncodingAttr::get(&context, 2);

  // 2. Define the shape and element type of the tensor
  llvm::SmallVector<int64_t> shape = {10, 20};
  mlir::Type elementType = builder.getF32Type();

  // 3. Create the RankedTensorType, passing our attribute as the 3rd argument (encoding)
  auto tensorType = mlir::RankedTensorType::get(shape, elementType, encodingAttr);

  // 4. Print the resulting type to the console
  llvm::outs() << "Generated Tensor Type: " << tensorType << "\n";

  return 0;
}
