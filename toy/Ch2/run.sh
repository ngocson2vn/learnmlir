#!/bin/bash

# print debug info
../bazel-bin/Ch2/toyc ./test/codegen.toy -emit=mlir -mlir-print-debuginfo -debug

# save irs
../bazel-bin/Ch2/toyc ./test/test1.toy -emit=mlir -mlir-print-debuginfo 2> test1.mlir
../bazel-bin/Ch2/toyc ./test/test2.toy -emit=mlir -mlir-print-debuginfo 2> test2.mlir
# ../bazel-bin/Ch2/toyc ./test/codegen.toy -emit=mlir -mlir-print-debuginfo 2> toy.mlir