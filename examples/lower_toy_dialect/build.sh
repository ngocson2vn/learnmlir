#!/bin/bash

set -e

export CMAKE_PREFIX_PATH=/data00/home/son.nguyen/workspace/learnmlir/llvm-project/build/lib/cmake/:$CMAKE_PREFIX_PATH
export UBSAN_LIBRARY_DIR=/data00/home/son.nguyen/workspace/learnmlir/llvm-project/build/lib/clang/16.0.0/lib/x86_64-unknown-linux-gnu

mkdir -p build && cd build

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
