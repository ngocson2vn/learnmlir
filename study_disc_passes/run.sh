#!/bin/bash

set -e

sudo rm -rf /opt/tiger/cores/*

./bazel-bin/main input.mlir

if [ "$?" != "0" ]; then
  find /opt/tiger/cores/
fi