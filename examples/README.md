# Setup
```Bash
# GCC
gcc --version
gcc (GCC) 10.1.0

# Submodules
git submodule add -f https://github.com/tensorflow/tensorflow.git examples/third_party/tensorflow
git config -f .gitmodules submodule.third_party/tensorflow.shallow true

git submodule update --init --recursive

cp -fv third_party/tensorflow/.bazelversion .
```

# Build
```Bash
./build.sh
```
