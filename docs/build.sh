#!/bin/bash

#############################################################################
# Prerequisites
#############################################################################
# cmake version 3.13.4
#
# apt install -y clang-11 lld-11 ninja-build ccache
# cd /usr/bin
# ln -s /usr/lib/llvm-11/bin/clang /usr/bin/clang
# ln -s /usr/lib/llvm-11/bin/clang++ /usr/bin/clang++


#############################################################################
# Build
#############################################################################
# mkdir llvm-project/build
# cp -vf docs/build.sh llvm-project/build
# cd llvm-project/build
# ./build.sh
#############################################################################

set -e

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
   -DLLVM_USE_SANITIZER="Address;Undefined"

cmake --build . --target check-mlir