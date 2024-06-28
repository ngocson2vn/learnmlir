#!/bin/bash

# print debug info
../bazel-bin/Ch2/toyc ./test/codegen.toy -emit=mlir -mlir-print-debuginfo -debug

# save ir
../bazel-bin/Ch2/toyc ./test/codegen.toy -emit=mlir -mlir-print-debuginfo 2> toy.mlir