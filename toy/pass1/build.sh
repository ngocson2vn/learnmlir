#!/bin/bash

set -e

CC=clang bazel build -s --compilation_mode=dbg --strip=never --verbose_failures --sandbox_debug :main
