include "mlir/Pass/PassBase.td"

def DiscLhloLegalizeRootsToParallelLoopsPass : Pass<"disc-lhlo-legalize-roots-to-parallel-loops", "mlir::func::FuncOp"> {
  let summary = "Legalize the roots of lmhlo.fusion to parallel loops.";
  let constructor = "createDiscLhloLegalizeRootsToParallelLoopsPass()";
  let options = [
    Option<"core_count_", "core-count", "int",
            /*default=*/"-1", "core count (e.g., SM count on NVIDIA GPU).">,
    Option<"cc_major_", "gpu-sm-cc-major", "int",
            /*default=*/"8", "gpu sm cc_major.">,
    Option<"cc_minor_", "gpu-sm-cc-minor", "int",
            /*default=*/"0", "gpu sm cc_minor.">,
  ];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::math::MathDialect",
    "gpu::GPUDialect",
    "disc_shape::DISCShapeDialect"
  ];
}
