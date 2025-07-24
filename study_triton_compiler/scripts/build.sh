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
# ln -s /usr/lib/llvm-11/bin/lld /usr/bin/lld


#############################################################################
# Build
#############################################################################
<<SCRIPT

mkdir llvm-project
cd llvm-project
git init
git remote add origin git@github.com:llvm/llvm-project.git
git fetch origin --depth 1 fdac4c4e92e5a83ac5e4fa6d1d2970c0c4df8fa8
git checkout FETCH_HEAD
cd -

cd llvm-project
rm -rf build && mkdir -v build
cd ..
yes | cp -vf docs/build.sh llvm-project/build
cd llvm-project/build
./build.sh

SCRIPT
#############################################################################

set -e

# cmake -G Ninja ../llvm \
#    -DLLVM_ENABLE_PROJECTS="mlir;compiler-rt" \
#    -DLLVM_BUILD_EXAMPLES=ON \
#    -DLLVM_TARGETS_TO_BUILD="Native;X86;NVPTX;AMDGPU" \
#    -DCMAKE_BUILD_TYPE=Release \
#    -DLLVM_ENABLE_ASSERTIONS=ON \
#    -DCMAKE_C_COMPILER=clang \
#    -DCMAKE_CXX_COMPILER=clang++ \
#    -DLLVM_ENABLE_LLD=ON \
#    -DLLVM_CCACHE_BUILD=ON \
#    -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
#    -DLLVM_INCLUDE_TESTS=OFF \
#    -DCOMPILER_RT_BUILD_SANITIZERS=ON \
#    -DLLVM_USE_SANITIZER="Address;Undefined"

#############################################################
# Debug mode
# -DCMAKE_CXX_FLAGS="-g -O0"
#############################################################

cmake -G Ninja ../llvm \
   -DCMAKE_CXX_FLAGS="-g -O0" \
   -DLLVM_ENABLE_PROJECTS="mlir;compiler-rt" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;X86;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON \
   -DLLVM_CCACHE_BUILD=ON \
   -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
   -DLLVM_INCLUDE_TESTS=OFF \
   -DCOMPILER_RT_BUILD_SANITIZERS=ON

cmake --build . #--target check-mlir