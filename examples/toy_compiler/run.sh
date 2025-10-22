#!/bin/bash

set -e

export CUDA_DIR=/usr/local/cuda-12.4
# build/main ./add_two_tensors.toy
build/main ./add_two_tensors.toy -debug
