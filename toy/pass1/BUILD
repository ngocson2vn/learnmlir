load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "toy_dialect_lib",
    hdrs = glob(["include/toy/*.h"]),
    includes = ["include"],
    srcs = [
      "mlir/toy_dialect.cc",
    ],
    deps = [
      "@llvm-project//mlir:Analysis",
      "@llvm-project//mlir:Parser",
      "@llvm-project//mlir:IR",
      "@llvm-project//mlir:SideEffectInterfaces",
      "@llvm-project//mlir:Transforms",
      "@llvm-project//mlir:Support",
    ]
)

cc_library(
  name = "traverse_lib",
  hdrs = glob(["include/traverse.h"]),
  srcs = [
    "mlir/traverse.cc"
  ],
  deps = [
    "@llvm-project//mlir:Pass",
    "@llvm-project//mlir:IR",
  ]
)

cc_binary(
    name = "main",
    srcs = [
      "main.cc",
    ],
    deps = [
      ":toy_dialect_lib",
      ":traverse_lib",
      "@llvm-project//mlir:AllPassesAndDialects",
      "@llvm-project//mlir:Pass",
      "@llvm-project//mlir:Dialect",
      "@llvm-project//mlir:Analysis",
      "@llvm-project//mlir:Parser",
      "@llvm-project//mlir:IR",
      "@llvm-project//mlir:SideEffectInterfaces",
      "@llvm-project//mlir:Transforms",
      "@llvm-project//mlir:MlirOptLib",
    ]
)
