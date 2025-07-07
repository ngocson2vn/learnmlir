#!/bin/bash

# bazel-bin/parse_dense_tensor/parse_dense_tensor

bazel-bin/lower_gpu_ops/lower_gpu_ops lower_gpu_ops/input.mlir

# bazel-bin/llvm_mlir_constant/llvm_mlir_constant

# ./bazel-bin/lower_lmhlo_ops/lower_lmhlo_ops
