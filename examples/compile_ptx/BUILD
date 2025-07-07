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
    "@org_tensorflow//tensorflow/compiler/xla:tensorflow_xla_bridge",
    "@org_tensorflow//tensorflow:tensorflow_framework_bridge",
    "@com_google_protobuf//:protobuf",
    "@com_google_protobuf//:protobuf_headers",
    "@com_google_absl//absl/strings",
  ]
)
