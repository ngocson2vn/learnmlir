# Study DISC Passes
## Dependencies
```Bash
git submodule add -f https://github.com/pai-disc/tensorflow.git third_party/tensorflow
git config -f .gitmodules submodule.third_party/tensorflow.shallow true
cd third_party/tensorflow
git reset --hard 1f7c9e80e6a9eb786d960b4b84133d645a36e39c

git submodule add -f https://github.com/pai-disc/torch-mlir.git third_party/torch-mlir
git config -f .gitmodules submodule.third_party/torch-mlir.shallow true
cd third_party/torch-mlir/
git reset --hard 500d11c7e52409fbdaeb53c742795f89ed6134d9

git submodule add -f https://github.com/oneapi-src/oneDNN.git third_party/oneDNN

# Only the first time
# cp sync.sh ~/workspace/BladeDISC/tao_compiler/
# cd ~/workspace/BladeDISC/tao_compiler
# ./sync.sh
```

This project demonstrates the MLIR lowering process:<br/>
- Create a toy dialect with only one op `toy.add`
- Lower `toy.add` to `arith.addf`
<br/>

## How to build
### Step 1: Build llvm-project
```Bash
cd learnmlir
mkdir llvm-project
cd llvm-project
git init
git remote add origin git@github.com:llvm/llvm-project.git
git fetch origin --depth 1 fdac4c4e92e5a83ac5e4fa6d1d2970c0c4df8fa8
git checkout FETCH_HEAD

mkdir -v build
cp -vf ../docs/build.sh build
cd build
./build.sh

# Verify
find | grep -E '.*\.a$' | sort -V
```
<br/>

### Step 2: Build this project
```Bash
./build.sh
```
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

# Reference
https://github.com/alibaba/BladeDISC