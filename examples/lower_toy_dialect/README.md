# Lower Toy Dialect
This project demonstrates the MLIR lowering process:<br/>
- Create a toy dialect with only one op `toy.add`
- Lower `toy.add` to `arith.addf`
<br/>

## How to build
### Step 1: Build llvm-project
```Bash
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
```Bash
./run.sh
```
<br/>

Lowered MLIR:
```mlir
module {
  func.func @main() {
    %cst = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %cst_0 = arith.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf64>
    %0 = arith.addf %cst, %cst_0 : tensor<2xf64>
    return
  }
}
```
