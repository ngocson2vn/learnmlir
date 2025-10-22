#!/bin/bash

set -e

export CUDA_ROOT=/usr/local/cuda-12.4
# build/main ./add_two_tensors.toy -debug

build/main ./add_two_tensors.toy
