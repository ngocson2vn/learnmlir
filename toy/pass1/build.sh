#!/bin/bash

set -e

export USE_BAZEL_VERSION=5.3.0
bazel version

# patch tensorflow
pushd ../third_party/tensorflow/
if [ ! -e tf.patch.done ]; then
  git apply ../tf.patch > tf.patch.done
fi
popd

CC=clang bazel build -s --compilation_mode=dbg --strip=never --verbose_failures --sandbox_debug --experimental_repo_remote_exec :main
