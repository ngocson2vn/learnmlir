#!/bin/bash

set -e

target_chapter="$1"

if ! [[ $target_chapter = @(Ch1|Ch2|Ch3|Ch4|Ch5|Ch6|Ch7) ]]; then
  echo "Build target \"$target_chapter\" is unsupported"
  exit 1
fi

build_root=../llvm-project/build
mlir_src_root=../llvm-project/mlir

echo "Start building $target_chapter"

if ! [ -e $target_chapter/.tblgen.done ]; then
  echo "Generate required definitions"

  # Ops.h.inc
  ${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-op-decls -o Ch2/include/toy/Ops.h.inc Ch2/include/toy/Ops.td

  # Ops.cpp.inc
  ${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-op-defs -o Ch2/include/toy/Ops.cpp.inc Ch2/include/toy/Ops.td

  # Dialect.h.inc
  ${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-dialect-decls -o Ch2/include/toy/Dialect.h.inc Ch2/include/toy/Ops.td

  # Dialect.cpp.inc
  ${build_root}/bin/mlir-tblgen -I ${mlir_src_root}/include/ -gen-dialect-defs -o Ch2/include/toy/Dialect.cpp.inc Ch2/include/toy/Ops.td

  touch $target_chapter/.tblgen.done
fi

# CC=clang bazel build //Ch2:test
CC=clang bazel build -s //$target_chapter:toyc --verbose_failures --sandbox_debug
