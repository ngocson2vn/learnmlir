load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "mlir_hlo_delegate",
    visibility = [
        "//visibility:private"
    ],
    deps = [
      "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:mlir_hlo",
      "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:lhlo",
    ]
)

cc_binary(
  name = "lower_lmhlo_ops",
  srcs = [
    "main.cc",
  ],
  deps = [
    ":mlir_hlo_delegate",
    "@llvm-project//mlir:AllPassesAndDialects",
    "@llvm-project//mlir:Analysis",
    "@llvm-project//mlir:IR",
  ]
)
