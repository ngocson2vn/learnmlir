#pragma once

#include <vector>
#include <string>

#include "llvm/Support/LogicalResult.h"

namespace llvm {

class Module;
class OptimizationLevel;
class Triple;

void initAllTargets();

LogicalResult linkExternLibs(llvm::Module *dstMod, const std::vector<std::string> &paths);

void optimizeLLVMModule(llvm::Module *llvmMod,
                        const llvm::OptimizationLevel &opt,
                        const std::string& triple,
                        const std::string& arch = "",
                        const std::string& features = "",
                        const std::vector<std::string>& flags = {},
                        bool enable_fp_fusion = false);

std::string translateLLVMIRToASM(llvm::Module* llvmMod,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags = {},
                                 bool enable_fp_fusion = false,
                                 bool isObject = false);

} // namespace llvm