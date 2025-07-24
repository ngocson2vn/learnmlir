set -e

#========================================================================
# llvm-project
#========================================================================
llvm_hash=$(cat ./cmake/llvm-hash.txt)
echo "llvm_hash=${llvm_hash}"
mkdir -p llvm-project
cd llvm-project
git init
git remote add origin git@github.com:llvm/llvm-project.git
git fetch origin --depth 1 ${llvm_hash}
git checkout FETCH_HEAD

rm -rf build
mkdir -v build
yes | cp -vf ../scripts/build.sh build/

pushd build/
./build.sh
popd
