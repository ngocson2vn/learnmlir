load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
  name = "lower_gpu_ops",
  srcs = [
    "main.cc",
  ],
  deps = [
    "@llvm-project//mlir:AllPassesAndDialects",
    "@llvm-project//mlir:Analysis",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:NVGPUDialect",
    "@llvm-project//mlir:NVVMDialect",
    "@llvm-project//mlir:Transforms",
    "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:all_passes",  # fixdeps: keep
    "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
    "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:transforms_gpu_passes",
    "@org_tensorflow//tensorflow/core:lib"
  ]
)
