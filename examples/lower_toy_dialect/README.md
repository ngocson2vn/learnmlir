# Lower Toy Dialect
This project demonstrates the MLIR lowering process:<br/>
- Create a toy dialect with only one op `toy.add`
- Lower `toy.add` to `arith.addf`
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
./build.sh
```
Both LLVM and MLIR libs will be built together.
<br/>

## How to run
input.mlir:
```mlir
module {
  func.func @main(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>) -> tensor<?xf64> {
    %0 = "toy.add"(%arg0, %arg1) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }
}
```
<br/>

```Bash
./run.sh
```
<br/>

Lowered MLIR:
```mlir
module {
  func.func @main(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>) -> tensor<?xf64> {
    %0 = arith.addf %arg0, %arg1 : tensor<?xf64>
    return %0 : tensor<?xf64>
  }
}
```
