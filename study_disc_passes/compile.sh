#!/bin/bash

set -e

export PRINT_IR_AFTER_PASS=true
export DISC_DEBUG_PRINT_FUSION_PARAMS=true

sudo rm -rf /opt/tiger/cores/*

./bazel-bin/main input.mlir > output.mlir

if [ -f ./debug.mlir ]; then
  find ./debug.mlir
fi

if [ -f output.mlir ]; then
  code output.mlir
fi
