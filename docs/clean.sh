#!/bin/bash

#========================================
# Copy this script to llvm-project/
# ./clean.sh
#========================================

set -e

rm -rf build && mkdir -v build
cp -vf ../docs/build.sh build
cd build
