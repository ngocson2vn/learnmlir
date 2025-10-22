include "mlir/Pass/PassBase.td"


def LowerMemRefToLLVMPass : Pass<"lower-memref-to-llvm", "ModuleOp"> {
  let summary = "Pass to lower memref to llvm";

  let constructor = "createLowerMemRefToLLVMPass()";
}
