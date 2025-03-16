load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
  name = "parse_dense_tensor",
  srcs = [
    "*.cc",
  ],
  deps = [
    "@llvm-project//mlir:Analysis",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:Transforms",
  ]
)
