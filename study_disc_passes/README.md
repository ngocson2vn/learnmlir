# Study DISC Passes
This project is for studying [BladeDISC](https://github.com/alibaba/BladeDISC) passes.<br/>
<br/>

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
<br/>

## How to build
```Bash
./build.sh
```
<br/>

## How to run
```Bash
./run.sh
```
<br/>
