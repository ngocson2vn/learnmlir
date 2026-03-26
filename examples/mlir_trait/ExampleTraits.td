include "mlir/IR/OpBase.td"

// Define the custom C++ trait
def VerifyMagicNumberTrait : NativeOpTrait<"VerifyMagicNumberTrait"> {
  // Tell TableGen where to find the C++ implementation
  let cppNamespace = "::mlir::example"; 
}
