#!/bin/bash

set -e

ROOT_DIR=$(pwd)
echo "==================================================="
echo "1. Update git submodules"
echo "==================================================="
if [[ ! -f ./.git_submodule_updated ]]; then
  git submodule update --init --recursive
  touch ./.git_submodule_updated
fi
echo "DONE"

echo
echo "==================================================="
echo "2. Build third_party/llvm-project/llvm"
echo "==================================================="
mkdir -p ${ROOT_DIR}/llvm-build
cmake -G Ninja -S third_party/llvm-project/llvm -B llvm-build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_ENABLE_PROJECTS="mlir;compiler-rt" \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="Native;X86;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_CCACHE_BUILD=ON \
  -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DCOMPILER_RT_BUILD_SANITIZERS=ON 1>/dev/null

cmake --build llvm-build/
echo "DONE"

echo
echo "==================================================="
echo "3. Build main.cpp"
echo "==================================================="
if ! git status | grep -q 'third_party/mlir-hlo'; then
  pushd third_party/mlir-hlo/
  git apply ../patches/mlir_hlo.patch
  echo "Patched third_party/mlir-hlo/"
  popd
fi

mkdir -p ${ROOT_DIR}/build

# -DLLVM_BINARY_DIR=${ROOT_DIR}/llvm-build \
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=${ROOT_DIR}/llvm-build/lib/cmake/mlir

# cmake --build build/ -v
cmake --build build/
echo "DONE"
