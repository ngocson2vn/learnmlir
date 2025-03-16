#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  // Create an MLIR context
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.disableMultithreading();

  // Define the IR string
  const std::string mlirInput = R"(
%cst_0 = arith.constant dense<"0x00000000CDCC4C3ECDCCCC3E9A99193FCDCC4C3F0000803F9A99993F3333B33FCDCCCC3F6666E63F00000040CDCC0C409A991940666626403333334000004040CDCC4C409A99594066666640333373400000804066668640CDCC8C40333393409A9999400000A0406666A640CDCCAC403333B3409A99B9400000C0406666C640CDCCCC403333D3409A99D9400000E0406666E640CDCCEC403333F3409A99F9400000004133330341666606419A990941CDCC0C410000104133331341666616419A991941CDCC1C410000204133332341666626419A992941CDCC2C410000304133333341666636419A993941CDCC3C410000404133334341666646419A994941CDCC4C410000504133335341666656419A995941CDCC5C410000604133336341666666419A996941CDCC6C410000704133337341666676419A997941CDCC7C41000080419A99814133338341CDCC844166668641000088419A99894133338B41CDCC8C4166668E41000090419A99914133339341CDCC944166669641000098419A99994133339B41CDCC9C4166669E410000A0419A99A1413333A341CDCCA4416666A6410000A8419A99A9413333AB41CDCCAC416666AE410000B0419A99B1413333B341CDCCB4416666B6410000B8419A99B9413333BB41CDCCBC416666BE410000C0419A99C1413333C341CDCCC4416666C6410000C841"> : tensor<1x1x126xf32>
)";

  // Parse the IR string into a ModuleOp
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(mlirInput, &context);
  if (!module) {
    llvm::errs() << "Failed to parse MLIR input\n";
    return 1;
  }

  // Traverse the module to find the arith.constant operation
  for (mlir::Operation &op : module->getBody()->getOperations()) {
    if (op.getName().getStringRef() == "arith.constant") {
      llvm::outs() << "Found arith.constant operation\n";

      // Get the 'value' attribute as a DenseElementsAttr
      mlir::DenseElementsAttr attr = op.getAttrOfType<mlir::DenseElementsAttr>("value");
      if (!attr) {
        llvm::errs() << "No dense attribute found\n";
        return 1;
      }

      // Verify the attribute is a tensor of f32
      if (!attr.getType().isa<mlir::RankedTensorType>() ||
          attr.getType().cast<mlir::RankedTensorType>().getElementType() != mlir::FloatType::getF32(&context)) {
        llvm::errs() << "Attribute is not a tensor of f32\n";
        return 1;
      }

      // Print the tensor shape
      auto tensorType = attr.getType().cast<mlir::RankedTensorType>();
      llvm::outs() << "Tensor shape: [";
      for (int64_t dim : tensorType.getShape()) {
        llvm::outs() << dim << " ";
      }
      llvm::outs() << "]\n";

      // Print the tensor elements
      llvm::outs() << "Tensor elements:\n";
      auto values = attr.getValues<float>(); // Get float values
      int index = 0;
      for (float value : values) {
        llvm::outs() << "Element[" << index << "] = " << value << "\n";
        index++;
      }
    }
  }

  return 0;
}
