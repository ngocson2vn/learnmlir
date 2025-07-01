load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
  name = "compile_ptx",
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
    "@org_tensorflow//tensorflow/core:framework",
    "@org_tensorflow//tensorflow/core:lib",
    "@org_tensorflow//tensorflow/core/platform:status",
    "@org_tensorflow//tensorflow/compiler/mlir:init_mlir",
    "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
    "@org_tensorflow//tensorflow/compiler/xla:xla_headers_lib",
    "@org_tensorflow//tensorflow/compiler/xla/service/gpu:gpu_asm_opts_util",
    "@org_tensorflow//tensorflow/compiler/xla/stream_executor/lib",
    "@org_tensorflow//tensorflow/compiler/xla/stream_executor/gpu:asm_compiler",
    "@org_tensorflow//tensorflow/compiler/mlir/tools/kernel_gen/transforms:gpu_passes",  # fixdeps: keep
    "@com_google_protobuf//:protobuf",
    "@com_google_protobuf//:protobuf_headers",
    "@com_google_absl//absl/strings",
  ]
)
