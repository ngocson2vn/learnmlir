#ifndef TFEXT_KERNEL_GEN_PASSES
#define TFEXT_KERNEL_GEN_PASSES

include "mlir/Pass/PassBase.td"

def SonyDebugPass: Pass<"af-sony-debug", "gpu::GPUModuleOp"> {
  let summary = "af-sony-debug";
  let constructor = "mlir::tfext_kernel_gen::createSonyDebugPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "gpu::GPUDialect",
    "mlir::LLVM::LLVMDialect",
  ];
}

#endif // TFEXT_KERNEL_GEN_PASSES