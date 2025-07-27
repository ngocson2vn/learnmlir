# Lower Toy Dialect
This project demonstrates the MLIR lowering process:<br/>
- Create a toy dialect with only one op `toy.add`
- Lower `toy.add` to `arith.addf`
<br/>

## Prerequisites
Install clang-11 and lld-11
```Bash
sudo apt install clang-11
sudo apt install lld-11

sudo ln -sf /usr/lib/llvm-11/bin/clang /usr/bin/clang
sudo ln -sf /usr/lib/llvm-11/bin/clang++ /usr/bin/clang++
sudo ln -sf /usr/lib/llvm-11/bin/lld /usr/bin/lld
sudo ln -sf /usr/lib/llvm-11/bin/ld.lld /usr/bin/ld.lld
```

## How to build
### Step 1: Clone llvm-project
```Bash
cd learnmlir
mkdir llvm-project
cd llvm-project
git init
git remote add origin git@github.com:llvm/llvm-project.git
git fetch origin --depth 1 fdac4c4e92e5a83ac5e4fa6d1d2970c0c4df8fa8
git checkout FETCH_HEAD

# mkdir -v build
# cp -vf ../docs/build.sh build
# cd build
# ./build.sh

# # Verify
# find | grep -E '.*\.a$' | sort -V
```
<br/>

### Step 2: Build this project
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
