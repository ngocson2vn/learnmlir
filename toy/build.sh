#!/bin/bash

set -e

target_chapter="$1"

if ! [[ $target_chapter = @(Ch1|Ch2|Ch3|Ch4|Ch5|Ch6|Ch7) ]]; then
  echo "Target \"$target_chapter\" is unsupported"
  exit 1
fi

# CC=clang bazel build //Ch2:test
echo "Start building $target_chapter"
CC=clang bazel build -s //$target_chapter:toyc --verbose_failures --sandbox_debug
