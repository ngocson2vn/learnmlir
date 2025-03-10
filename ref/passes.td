#ifndef TFEXT_KERNEL_GEN_PASSES
#define TFEXT_KERNEL_GEN_PASSES

include "mlir/Pass/PassBase.td"

def BatchComputeFusion: Pass<"af-batch-compute", "ModuleOp"> {
  let summary = "af-batch-compute";
  let constructor = "mlir::tfext_kernel_gen::createBatchComputeFusionPass()";
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect",
    "mlir::tf_executor::TensorFlowExecutorDialect",
    "AffineDialect",
    "gpu::GPUDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif // TFEXT_KERNEL_GEN_PASSES