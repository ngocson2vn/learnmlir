#!/bin/bash

for f in $(find build -mindepth 1 -maxdepth 1 | grep -v llvm-project)
do
  rm -rf $f
done

rm -rf build/llvm-project/build