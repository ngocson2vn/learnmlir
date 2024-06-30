#!/bin/bash

set -e

CC=clang bazel build -s --verbose_failures --sandbox_debug :main
