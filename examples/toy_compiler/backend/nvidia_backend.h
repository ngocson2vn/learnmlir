#pragma once

#include "llvm/Support/LogicalResult.h"

namespace llvm {
class Module;
}

namespace nvidia {

static const char* kTriple = "nvptx64-nvidia-cuda";

llvm::LogicalResult linkLibdevice(llvm::Module *llvmMod, int capability);

std::string translateLLVMIRToPTX(llvm::Module* llvmMod,
                                 const std::string& arch,
                                 const std::string& features,
                                 const std::vector<std::string> &flags = {},
                                 bool enable_fp_fusion = false,
                                 bool isObject = false);

std::string translatePTXtoCUBIN(const std::string& ptxFile,
                                const std::string& arch,
                                const std::string& outputCubinFile);

} // namespace nvidia
