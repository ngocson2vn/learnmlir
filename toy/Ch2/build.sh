#!/bin/bash

set -e

build_root=../../llvm-project/build
mlir_src_root=../../llvm-project/mlir

echo "Start building chapter 2"

if [ ! -e ./.tblgen.done ]; then
  echo "Generate required definitions"

  # Ops.h.inc
  ${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-op-decls -o ./include/toy/Ops.h.inc ./include/toy/Ops.td

  # Ops.cpp.inc
  ${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-op-defs -o ./include/toy/Ops.cpp.inc ./include/toy/Ops.td

  # Dialect.h.inc
  ${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-dialect-decls -o ./include/toy/Dialect.h.inc ./include/toy/Ops.td

  # Dialect.cpp.inc
  ${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-dialect-defs -o ./include/toy/Dialect.cpp.inc ./include/toy/Ops.td

  touch ./.tblgen.done
fi

# CC=clang bazel build //Ch2:test

COPTS="-O0,-g,-fno-inline,-UNDEBUG"
SRC_FILES=+toyc.cpp
SRC_FILES=${SRC_FILES},+llvm/lib/Support/CommandLine.cpp
CC=clang++ bazel build -s :toyc --per_file_copt=${SRC_FILES}@${COPTS} --strip=never --verbose_failures --sandbox_debug --experimental_repo_remote_exec
