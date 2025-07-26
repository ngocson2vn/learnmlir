wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo apt install -y lsb-release wget software-properties-common gnupg

LLVM_VERSION=17
sudo ./llvm.sh $LLVM_VERSION

# MLIR
sudo apt install -y mlir-17-tools