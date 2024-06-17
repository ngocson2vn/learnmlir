#!/bin/bash

build_root=../llvm-project/build
mlir_src_root=../llvm-project/mlir

################################
# Ch2
################################
# Ops.h.inc
${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-op-decls -o Ch2/include/toy/Ops.h.inc Ch2/include/toy/Ops.td

# Ops.cpp.inc
${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-op-defs -o Ch2/include/toy/Ops.cpp.inc Ch2/include/toy/Ops.td

# Dialect.h.inc
${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-dialect-decls -o Ch2/include/toy/Dialect.h.inc Ch2/include/toy/Ops.td

# Dialect.cpp.inc
${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-dialect-defs -o Ch2/include/toy/Dialect.cpp.inc Ch2/include/toy/Ops.td
