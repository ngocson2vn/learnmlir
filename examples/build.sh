#!/bin/bash

set -e

# CC=clang bazel build //Ch2:test
CC=clang bazel build -s //parse_dense_tensor:parse_dense_tensor --verbose_failures --sandbox_debug --experimental_repo_remote_exec