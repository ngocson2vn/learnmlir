#!/bin/bash

set -e

# debug
../bazel-bin/Ch3/toyc ./test/test1.toy -emit=mlir -opt -mlir-print-debuginfo -debug

# save ir
../bazel-bin/Ch3/toyc ./test/test1.toy -emit=mlir -opt -mlir-print-debuginfo 2> test1.mlir
echo "Generated file test1.mlir"
