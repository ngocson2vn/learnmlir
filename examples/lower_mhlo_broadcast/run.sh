#!/bin/bash

set -e

./build/main ./module.mlir

if which code >/dev/null; then
  code ./lowering.mlir
fi
