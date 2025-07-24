#!/bin/bash

set -e

ROOT_DIR=$(pwd)
echo "ROOT_DIR=${ROOT_DIR}"

#========================================================
# Prerequisites
#========================================================
# llvm-project
llvm_hash=fdac4c4e92e5a83ac5e4fa6d1d2970c0c4df8fa8
echo "llvm_hash=${llvm_hash}"
mkdir -p llvm-project
cd llvm-project
git init
git remote add origin git@github.com:llvm/llvm-project.git
git fetch origin --depth 1 ${llvm_hash}
git checkout FETCH_HEAD
cd ..

rm -rf llvm-project/build
mkdir -v llvm-project/build
yes | cp -vf ../../docs/build.sh llvm-project/build

pushd llvm-project/build
./build.sh
popd


export CMAKE_PREFIX_PATH=${ROOT_DIR}/llvm-project/build/lib/cmake/:$CMAKE_PREFIX_PATH
export UBSAN_LIBRARY_DIR=${ROOT_DIR}/llvm-project/build/lib/clang/16.0.0/lib/x86_64-unknown-linux-gnu

cd ${ROOT_DIR}/
mkdir -p build
cd build

echo
echo "==================================================="
echo "Generate ninja build file"
echo "==================================================="
cmake -G Ninja ..

echo
echo "==================================================="
echo "Run ninja build"
echo "==================================================="
cmake --build . -- -v
