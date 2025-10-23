#include <fstream>
#include <filesystem>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#include "utils.h"
#include "cuda_utils.h"
#include "llvm_utils.h"
#include "nvidia_backend.h"

namespace fs = std::filesystem;

namespace nvidia {

// Ensure that CUDA_DIR, ptxas, and libdevice exist
static bool ok = []() -> bool {
  auto res = cuda::getCudaRoot();
  if (!res.ok()) {
    llvm::errs() << res.error_message() << "\n";
    std::abort();
  }

  auto ptxasPath = cuda::getPtxasPath();
  if (!fs::exists(ptxasPath)) {
    llvm::errs() << ptxasPath << " does not exists!\n";
    std::abort();
  }

  std::string libdevice = cuda::getLibdevice();
  if (!fs::exists(libdevice)) {
    llvm::errs() << libdevice << " does not exists!\n";
    std::abort();
  }

  return true;
}();

llvm::LogicalResult linkLibdevice(llvm::Module *llvmMod, int capability) {
  std::string arch = std::string("sm_").append(std::to_string(capability));
  std::string features = std::string("+ptx").append(std::to_string(capability));
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(llvm::Triple(kTriple), error);
  if (!target) {
    llvm::errs() << "target lookup error: " + error << "\n";
    std::terminate();
  }

  llvm::TargetOptions opt;
  // Target machine is only used to create the data layout.
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      llvm::Triple(kTriple), arch, features, opt, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::None));

  // set data layout
  llvmMod->setDataLayout(machine->createDataLayout());

  // set_nvvm_reflect_ftz
  auto& ctx = llvmMod->getContext();
  llvm::Type* i32 = llvm::Type::getInt32Ty(ctx);
  auto* mdFour = llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(i32, 4));
  auto* mdName = llvm::MDString::get(ctx, "nvvm-reflect-ftz");
  auto* mdOne = llvm::ConstantAsMetadata::get(llvm::ConstantInt::getSigned(i32, 1));
  auto* reflect = llvm::MDNode::get(ctx, {mdFour, mdName, mdOne});
  llvmMod->addModuleFlag(reflect);

  std::vector<std::string> paths{cuda::getLibdevice()};

  return linkExternLibs(llvmMod, paths);
}

std::string translateLLVMIRToPTX(llvm::Module* llvmMod,
                                 const std::string& arch,
                                 const std::string& features,
                                 const std::vector<std::string> &flags,
                                 bool enable_fp_fusion,
                                 bool isObject) {
  return llvm::translateLLVMIRToASM(
    llvmMod,
    nvidia::kTriple,
    arch,
    features, 
    flags,
    enable_fp_fusion,
    isObject
  );
}

std::string translatePTXtoCUBIN(const std::string& ptxFile, 
                                const std::string& arch,
                                const std::string& outputCubinFile) {
  std::string ptxasPath = cuda::getPtxasPath();
  std::string cmd = ptxasPath
                    .append(" -lineinfo -suppress-debug-info")
                    .append(" --fmad=false -v")
                    .append(" --gpu-name=").append(arch)
                    .append(" -o ").append(outputCubinFile)
                    .append(" ").append(ptxFile);
  std::string cubinStr;
  std::string stdoutOutput, stderrOutput;
  auto res = toy::utils::runCommand(cmd, stdoutOutput, stderrOutput);
  if (!res.ok()) {
    llvm::errs() << "ptxas failed\n\n";
    llvm::errs() << "stderr:\n" << stderrOutput << "\n";
    return std::string();
  }

  return toy::utils::readFile(outputCubinFile);
}

} // namespace nvidia