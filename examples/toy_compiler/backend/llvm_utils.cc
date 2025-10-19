#include <unordered_set>

#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/IR/Constants.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"

#include "utils.h"
#include "llvm_utils.h"

namespace {

std::unique_ptr<llvm::TargetMachine>
createTargetMachine(llvm::Module *module, std::string proc,
                    bool enable_fp_fusion, const std::string &features) {
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  llvm::TargetOptions opt;
  bool disableLLVMOpt = toy::utils::getBoolEnv("DISABLE_LLVM_OPT");
  if (enable_fp_fusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  opt.MCOptions.AsmVerbose = true;
  opt.MCOptions.PreserveAsmComments = true;
  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt,
      disableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive)};
  return machine;
}

bool processPhiStruct(llvm::PHINode *phiNode) {
  llvm::StructType *STy = llvm::dyn_cast<llvm::StructType>(phiNode->getType());
  if (!STy)
    return false;
  llvm::IRBuilder<> builder(phiNode);
  unsigned numOperands = phiNode->getNumIncomingValues();
  unsigned numScalarEl = STy->getNumElements();
  llvm::Value *newStruct = llvm::UndefValue::get(STy);
  builder.SetInsertPoint(phiNode->getParent()->getFirstNonPHIIt());
  llvm::IRBuilderBase::InsertPoint insertInsertPt = builder.saveIP();
  for (unsigned i = 0; i < numScalarEl; i++) {
    builder.SetInsertPoint(phiNode);
    llvm::PHINode *newPhiNode =
        builder.CreatePHI(STy->getElementType(i), numOperands);
    for (unsigned j = 0; j < numOperands; ++j) {
      llvm::Value *operand = phiNode->getIncomingValue(j);
      builder.SetInsertPoint(phiNode->getIncomingBlock(j)->getTerminator());
      newPhiNode->addIncoming(builder.CreateExtractValue(operand, i),
                              phiNode->getIncomingBlock(j));
    }
    builder.restoreIP(insertInsertPt);
    newStruct = builder.CreateInsertValue(newStruct, newPhiNode, i);
    insertInsertPt = builder.saveIP();
  }
  phiNode->replaceAllUsesWith(newStruct);
  return true;
}

bool runOnFunction(llvm::Function &F) {
  bool Changed = false;
  llvm::SmallVector<llvm::PHINode *> PhiNodes;
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &inst : BB) {
      if (llvm::PHINode *phiNode = llvm::dyn_cast<llvm::PHINode>(&inst)) {
        Changed |= processPhiStruct(phiNode);
        continue;
      }
      break;
    }
  }
  return Changed;
}

}

namespace llvm {

void initTargets() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  });
}

struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    bool b = runOnFunction(F);
    return b ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  static StringRef name() { return "BreakStructPhiNodesPass"; }
};

llvm::LogicalResult linkExternLibs(llvm::Module *dstMod, const std::vector<std::string> &paths) {
  if (paths.empty())
    return failure();

  llvm::LLVMContext &ctx = dstMod->getContext();
  llvm::Linker linker(*dstMod);
  for (const std::string &path : paths) {
    llvm::SMDiagnostic err;
    std::unique_ptr<llvm::Module> libMod = llvm::parseIRFile(path, err, ctx);
    if (!libMod) {
      std::string message = "Failed to parse library at " + path;
      llvm::errs() << message << "\n";
      return failure();
    }
    libMod->setTargetTriple(llvm::Triple(dstMod->getTargetTriple()));
    libMod->setDataLayout(dstMod->getDataLayout());

    std::unordered_set<std::string> externalFns;
    for (llvm::Function &fn : libMod->functions()) {
      if (!fn.isDeclaration())
        externalFns.insert(fn.getName().str());
    }

    if (linker.linkInModule(std::move(libMod),
                            llvm::Linker::Flags::LinkOnlyNeeded)) {
      std::string message = "Failed to link library at " + path;
      llvm::errs() << message << "\n";
      return failure();
    }

    // Mark linked-in functions as internal because backends use external
    // linkage as a signifier of kernel functions.
    for (llvm::Function &fn : dstMod->functions()) {
      if (externalFns.count(fn.getName().str())) {
        fn.setLinkage(llvm::GlobalValue::InternalLinkage);
      }
    }
  }

  return success();
}

void optimizeLLVMModule(llvm::Module *llvmMod,
                        const llvm::OptimizationLevel &opt,
                        const std::string& triple,
                        const std::string& arch,
                        const std::string& features,
                        const std::vector<std::string>& flags,
                        bool enable_fp_fusion) {
  if (toy::utils::getBoolEnv("DISABLE_LLVM_OPT"))
    return;
  // Check to see if we are passing a list of flags to disable
  // optimizations.
  auto flagList = toy::utils::getStrEnv("DISABLE_LLVM_OPT");
  if (!flagList.empty()) {
    auto options = llvm::cl::getRegisteredOptions();
    llvm::SmallVector<StringRef, 3> split;
    StringRef(flagList.c_str()).split(split, ',');
    for (auto flag : split) {
      auto optIt = options.find(flag);
      if (optIt != options.end()) {
        auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
        *optPtr = true;
      }
    }
  }
  using namespace llvm;
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  if (arch.empty()) {
    Triple tp = Triple(triple);
    llvm::TargetLibraryInfoImpl TLII(tp);
    TLII.disableAllFunctions();
    fam.registerPass([TLII = std::move(TLII)] {
      return llvm::TargetLibraryAnalysis(TLII);
    });
  }

  PassInstrumentationCallbacks *instrCbPtr = nullptr;
  PassInstrumentationCallbacks passInstrCb;
  StandardInstrumentations standardInstr(llvmMod->getContext(),
                                          /*DebugLogging*/ true);
  if (toy::utils::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
    auto optMap = llvm::cl::getRegisteredOptions();
    auto optIt = optMap.find("print-after-all");
    if (optIt != optMap.end()) {
      auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
      *optPtr = true;
    }
    standardInstr.registerCallbacks(passInstrCb, &mam);
    instrCbPtr = &passInstrCb;
  }

  PipelineTuningOptions tuningOptions;
  tuningOptions.LoopUnrolling = true;
  tuningOptions.LoopInterleaving = true;
  tuningOptions.LoopVectorization = true;
  // TODO: currently we run SLP vectorizer with an empty target machine.
  // This cause the vectorizer to create larger vector which could be bad.
  // Disabling it would currently cause regressions as this pass also
  // applies some scheduling that helps performance in some cases. We
  // should work on using NVPTX target instead and address the performance
  // regressions with some scheduling solution.
  tuningOptions.SLPVectorization = true;

  std::string pluginFile =
      toy::utils::getStrEnv("LLVM_PASS_PLUGIN_PATH");

  // We don't pass the targetMachine to the LLVM-IR pass builder, unless
  // `arch` is specified.
  //
  // Don't set target machine in LLVM pass builder when using LLVM IR
  // level plugins. LLVM IR level plugin passes typically want to insert
  // calls to externally generated code (i.e. precompile a Cuda/Hip kernel
  // with Clang and then insert a call to it within an instrumentation
  // pass) setting the targetMachine value here can can cause a mismatch
  // in the target machine between the MLIR and Clang generated kernels
  // and break the lowering of some target specific intrinsics.
  std::unique_ptr<TargetMachine> targetMachine = nullptr;
  if (!arch.empty() && pluginFile.empty())
    targetMachine =
        createTargetMachine(llvmMod, arch, enable_fp_fusion, features);
  PassBuilder pb(/*targetMachine=*/targetMachine.get(), tuningOptions,
                  std::nullopt, instrCbPtr);

  if (!pluginFile.empty()) {
    // TODO: Add some logging here that we inserted a pass into the LLVM
    // pass pipeline
    auto passPlugin = llvm::PassPlugin::Load(pluginFile);
    if (!passPlugin) {
      llvm::Error Err = passPlugin.takeError();
      std::string ErrMsg = "Pass Plugin Error: " + llvm::toString(std::move(Err));
      llvm::errs() << ErrMsg << "\n";
      std::terminate();
    }
    passPlugin->registerPassBuilderCallbacks(pb);
  }

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  ModulePassManager mpm;
  pb.registerVectorizerStartEPCallback(
      [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
        // Triton generates large structure of scalars which may pessimise
        // optimizations, we run a pass to break up phi of struct to make
        // sure all the struct are removed for the following passes.
        fpm.addPass(BreakStructPhiNodesPass());
        fpm.addPass(InstCombinePass());
      });
  bool enableAddressSanitizer =
      toy::utils::getBoolEnv("TRITON_ENABLE_ASAN");
  if (enableAddressSanitizer) {
    AddressSanitizerOptions Opts;
    mpm.addPass(AddressSanitizerPass(Opts));
  }
  mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
  mpm.run(*llvmMod, mam);
}

std::string translateLLVMIRToASM(llvm::Module* llvmMod,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags,
                                 bool enable_fp_fusion,
                                 bool isObject) {
  using namespace mlir;
  llvm::Module& module = *llvmMod;

  // options
  auto options = llvm::cl::getRegisteredOptions();
  for (std::string flag : flags) {
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  }
  if (toy::utils::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
    auto optIt = options.find("print-after-all");
    if (optIt != options.end()) {
      auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
      *optPtr = true;
    }
  }
  bool disableLLVMOpt = toy::utils::getBoolEnv("DISABLE_LLVM_OPT");
  if (!disableLLVMOpt) {
    // Check to see if we are passing a list of flags to disable optimizations.
    auto flagList = toy::utils::getStrEnv("DISABLE_LLVM_OPT");
    if (!flagList.empty()) {
      llvm::SmallVector<StringRef, 3> split;
      StringRef(flagList.c_str()).split(split, ',');
      for (auto flag : split) {
        auto optIt = options.find(flag);
        if (optIt != options.end()) {
          auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
          *optPtr = true;
        }
      }
    }
  }

  // inline everything
  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());

  const bool enabledTiming = toy::utils::getBoolEnv("LLVM_ENABLE_TIMING");
  if (enabledTiming) {
    llvm::TimePassesIsEnabled = true;
    llvm::TimePassesPerRun = true;
  }

  pm.run(module);

  SmallString<0> timePassesStr;
  llvm::raw_svector_ostream reportStream(timePassesStr);

  if (enabledTiming) {
    reportAndResetTimings(&reportStream);
    llvm::dbgs() << reportStream.str();
    timePassesStr.clear();
  }
  // module->print(llvm::outs(), nullptr);

  // create machine
  module.setTargetTriple(llvm::Triple(triple));
  auto machine = createTargetMachine(&module, proc, enable_fp_fusion, features);
  // set data layout
  module.setDataLayout(machine->createDataLayout());
  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager pass;
    // emit
    auto fileType = isObject ? llvm::CodeGenFileType::ObjectFile
                             : llvm::CodeGenFileType::AssemblyFile;
    machine->addPassesToEmitFile(pass, pstream, nullptr, fileType);
    pass.run(module);

    if (enabledTiming) {
      reportAndResetTimings(&reportStream);
      llvm::dbgs() << reportStream.str();
      timePassesStr.clear();
    }
  }
  return result;
}

} // namespace llvm
