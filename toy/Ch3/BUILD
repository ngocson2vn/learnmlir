load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
  name = "toylib",
  hdrs = glob(["include/toy/*.h", "include/toy/*.inc"])
)

cc_binary(
    name = "toyc",
    srcs = [
      "toyc.cpp",
      "parser/AST.cpp",
      "mlir/MLIRGen.cpp",
      "mlir/Dialect.cpp",
      "mlir/ToyCombine.inc",
      "mlir/ToyCombine.cpp",
    ],
    copts = [
      "-ICh3/include"
    ],
    deps = [
      ":toylib",
      "@llvm-project//mlir:Analysis",
      "@llvm-project//mlir:Parser",
      "@llvm-project//mlir:IR",
      "@llvm-project//mlir:SideEffectInterfaces",
      "@llvm-project//mlir:Transforms",
    ]
)
