#!/bin/bash

set -e

# CC=clang bazel build //Ch2:test
# CC=clang bazel build -s //parse_dense_tensor:parse_dense_tensor --verbose_failures --sandbox_debug --experimental_repo_remote_exec

# CC=clang bazel build //lower_gpu_ops:lower_gpu_ops --compilation_mode=dbg --strip=never --verbose_failures --sandbox_debug --experimental_repo_remote_exec

# CC=clang bazel build //llvm_mlir_constant:llvm_mlir_constant --compilation_mode=dbg --strip=never --verbose_failures --sandbox_debug --experimental_repo_remote_exec

CC=clang bazel build //compile_ptx:compile_ptx --compilation_mode=dbg --config=cuda --strip=never --verbose_failures --sandbox_debug --experimental_repo_remote_exec