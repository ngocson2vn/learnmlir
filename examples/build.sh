#!/bin/bash

set -e

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.4/}
export CUDA_TOOLKIT_PATH="${CUDA_HOME}"
export TF_CUDA_HOME=${CUDA_HOME} # for cuda_supplement_configure.bzl
export TF_CUDA_PATHS="${CUDA_HOME},${HOME}/.cache/cudnn/"
export TF_CUDA_COMPUTE_CAPABILITIES="7.5,8.0,8.6"
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH

BAZEL_DISK_CACHE_DIR=~/.cache/bazel/_bazel_son.nguyen/cache/

# CC=gcc bazel build //Ch2:test
# CC=gcc bazel build -s //parse_dense_tensor:parse_dense_tensor --verbose_failures --sandbox_debug --experimental_repo_remote_exec

CC=gcc bazel build //lower_gpu_ops:lower_gpu_ops \
--disk_cache=${BAZEL_DISK_CACHE_DIR} \
--compilation_mode=dbg --strip=never --verbose_failures --sandbox_debug --experimental_repo_remote_exec

# CC=gcc bazel build //llvm_mlir_constant:llvm_mlir_constant --compilation_mode=dbg --strip=never --verbose_failures --sandbox_debug --experimental_repo_remote_exec

# Note: need to fix tensorflow packages' visibility
# CC=gcc bazel build //compile_ptx:compile_ptx \
# --disk_cache=${BAZEL_DISK_CACHE_DIR} \
# --compilation_mode=dbg --config=cuda --strip=never --verbose_failures --sandbox_debug --experimental_repo_remote_exec

# CC=gcc bazel build //lower_lmhlo_ops:lower_lmhlo_ops \
# --disk_cache=${BAZEL_DISK_CACHE_DIR} \
# --compilation_mode=dbg --strip=never --verbose_failures --sandbox_debug --experimental_repo_remote_exec
