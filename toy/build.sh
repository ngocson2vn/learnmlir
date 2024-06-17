#!/bin/bash

# CC=clang bazel build //Ch2:test
CC=clang bazel build -s //Ch2:toyc --verbose_failures --sandbox_debug
