#!/bin/bash

set -e

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4/}
export CUDA_TOOLKIT_PATH="${CUDA_HOME}"
export TF_CUDA_HOME=${CUDA_HOME} # for cuda_supplement_configure.bzl
export TF_CUDA_PATHS="${CUDA_HOME},${HOME}/.cache/cudnn/"
export TF_CUDA_COMPUTE_CAPABILITIES="7.5,8.0,8.6"
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Patch tensorflow
if [ ! -f third_party/tensorflow/patch.done ]; then
  pushd third_party/tensorflow/
  git apply ../patches/tf.patch > patch.done
  popd
fi


SRC_FILE_LIST=+main.cpp
SRC_FILE_LIST=${SRC_FILE_LIST},+mlir/disc/transforms/.*.cc
# SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/common_runtime/executor.cc

CC=clang bazel --output_user_root=./build build //:main \
--disk_cache=./build/cache \
--compilation_mode=opt --config=cuda \
--linkopt=-g --per_file_copt=${SRC_FILE_LIST}@-O0,-g,-fno-inline \
--define build_with_onednn_v2=true \
--strip=never --verbose_failures --sandbox_debug --experimental_repo_remote_exec

if [ "$?" == "0" ]; then
  ln -sf bazel-bin/main .
fi