#!/bin/bash

set -e

sudo rm -rf /opt/tiger/cores/*

./bazel-bin/main input.mlir > output.mlir

find output.mlir