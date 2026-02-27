# Triton Encodings
This project is for studying [Triton Encodings](https://github.com/triton-lang/triton/blob/eaaa75cf5daf498a4aab29f135fa07977e318fb6/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td)<br/>
<br/>

## Prerequisites
Install cmake version >=3.30.0
```Bash
apt install libssl-dev

# create a `cmake` directory and extract cmake into there
# build cmake in there, and install to prefix

PREFIX=/usr

wget -SL https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0.tar.gz
mkdir -p cmake-3.30.0
tar -xvf cmake-3.30.0.tar.gz --strip-components=1 -C cmake-3.30.0

cd cmake-3.30.0
./bootstrap --prefix=$PREFIX --parallel=`nproc`
nice -n20 make -j`nproc`

sudo nice -n20 make install
```

Install clang-17 and lld-17
```Bash
sudo apt install clang-17
sudo apt install lld-17

sudo ln -sf /usr/lib/llvm-17/bin/clang /usr/bin/clang
sudo ln -sf /usr/lib/llvm-17/bin/clang++ /usr/bin/clang++
sudo ln -sf /usr/lib/llvm-17/bin/ld.lld /usr/bin/ld.lld
```

## How to build
```Bash
git submodule update --init --recursive
make
```
Both LLVM and MLIR libs will be built together.
<br/>


