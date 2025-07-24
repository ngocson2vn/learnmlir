#include "RegisterTritonDialects.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Constants.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
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
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerOptions.h"

#include "triton/Tools/Sys/GetEnv.hpp"

#include <string>
#include <unordered_set>

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"),
    llvm::cl::init("-"));

namespace llvm {
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "BreakStructPhiNodesPass"; }
};
} // namespace llvm

void linkExternLibs(llvm::Module *dstMod, const std::vector<std::string> &paths) {
  if (paths.empty())
    return;

  llvm::LLVMContext &ctx = dstMod->getContext();
  llvm::Linker linker(*dstMod);
  for (const std::string &path : paths) {
    llvm::SMDiagnostic err;
    std::unique_ptr<llvm::Module> libMod = llvm::parseIRFile(path, err, ctx);
    if (!libMod) {
      std::string message = "Failed to parse library at " + path;
      llvm::errs() << message << "\n";
      std::terminate();
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
      std::terminate();
    }

    // Mark linked-in functions as internal because backends use external
    // linkage as a signifier of kernel functions.
    for (llvm::Function &fn : dstMod->functions()) {
      if (externalFns.count(fn.getName().str())) {
        fn.setLinkage(llvm::GlobalValue::InternalLinkage);
      }
    }
  }
}

std::unique_ptr<llvm::TargetMachine>
createTargetMachine(llvm::Module *module, std::string proc,
                    bool enable_fp_fusion, const std::string &features) {
  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(
      module->getTargetTriple().str(), error);
  llvm::TargetOptions opt;
  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
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

void optimizeLLVMModule(llvm::Module *mod, const llvm::OptimizationLevel &opt,
    std::string arch = "", std::string features = "", std::vector<std::string> flags = {},
    bool enable_fp_fusion = false) {
  if (mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT"))
    return;
  // Check to see if we are passing a list of flags to disable
  // optimizations.
  auto flagList = mlir::triton::tools::getStrEnv("DISABLE_LLVM_OPT");
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
    llvm::TargetLibraryInfoImpl TLII;
    TLII.disableAllFunctions();
    fam.registerPass([TLII = std::move(TLII)] {
      return llvm::TargetLibraryAnalysis(TLII);
    });
  }

  PassInstrumentationCallbacks *instrCbPtr = nullptr;
  PassInstrumentationCallbacks passInstrCb;
  StandardInstrumentations standardInstr(mod->getContext(),
                                          /*DebugLogging*/ true);
  if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
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
      mlir::triton::tools::getStrEnv("LLVM_PASS_PLUGIN_PATH");

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
        createTargetMachine(mod, arch, enable_fp_fusion, features);
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
      mlir::triton::tools::getBoolEnv("TRITON_ENABLE_ASAN");
  if (enableAddressSanitizer) {
    AddressSanitizerOptions Opts;
    mpm.addPass(AddressSanitizerPass(Opts));
  }
  mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
  mpm.run(*mod, mam);
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Triton compiler\n");

  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

  MLIRContext context(registry);

  // Load the input MLIR file
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return 1;
  }

  // Parse the input MLIR
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error parsing input file\n";
    return 1;
  }

  // Set up the pass manager
  PassManager pm(&context);

  // 
  // make_ttgir
  // 
  int capability = 86;
  std::string targetStr = std::string("cuda:").append(std::to_string(capability));
  pm.addPass(mlir::triton::createConvertTritonToTritonGPU({targetStr, 4, 32, 1}));
  pm.addPass(mlir::triton::gpu::createTritonGPUCoalesce());
  pm.addPass(mlir::triton::gpu::createTritonGPUF32DotTC());

  auto cluster_info = mlir::triton::nvidia_gpu::ClusterInfo();
  pm.addPass(mlir::triton::nvidia_gpu::createTritonNvidiaGPUPlanCTAPass(&cluster_info));

  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeThreadLocality());
  pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmul());
  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands({true}));
  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());

  pm.addPass(mlir::triton::nvidia_gpu::createTritonNvidiaGPUOptimizeDescriptorEncodingPass());

  pm.addPass(mlir::triton::createTritonLoopAwareCSE());

  pm.addPass(mlir::triton::gpu::createTritonGPUFuseNestedLoops());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(mlir::triton::createTritonLoopInvariantCodeMotion());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());

  int num_stages = 3;
  pm.addPass(mlir::createNVGPUWarpSpecialization({num_stages, true}));

  pm.addPass(mlir::triton::gpu::createTritonGPUAssignLatencies({num_stages}));
  pm.addPass(mlir::triton::gpu::createTritonGPUScheduleLoops());
  pm.addPass(mlir::triton::gpu::createTritonGPUPipeline({num_stages, true}));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::triton::createTritonLoopAwareCSE());
  
  pm.addPass(mlir::triton::gpu::createTritonGPUPrefetch());
  pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands({true}));
  pm.addPass(mlir::triton::gpu::createTritonGPUCoalesceAsyncCopy());

  pm.addPass(mlir::triton::nvidia_gpu::createTritonNvidiaGPUOptimizeTMemLayoutsPass());

  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());

  pm.addPass(mlir::triton::nvidia_gpu::createTritonNvidiaGPUInterleaveTMemPass());

  pm.addPass(mlir::triton::gpu::createTritonGPUReduceDataDuplication());
  pm.addPass(mlir::triton::gpu::createTritonGPUReorderInstructions());
  pm.addPass(mlir::triton::createTritonLoopAwareCSE());
  pm.addPass(mlir::createSymbolDCEPass());

  {
    mlir::triton::nvidia_gpu::TritonGPUFenceInsertionOptions options;
    options.computeCapability = capability;
    pm.addPass(mlir::triton::nvidia_gpu::createTritonGPUFenceInsertion(options));
  }

  pm.addPass(mlir::triton::nvidia_gpu::createTritonNvidiaGPUMMALoweringPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCanonicalizerPass());
  
  
  // 
  // make_llir
  //
  int ptxVersion = 128;
  pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
  pm.addPass(mlir::triton::gpu::createTritonGPUAllocateWarpGroups());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::triton::gpu::createAllocateSharedMemory());
  pm.addPass(mlir::triton::nvidia_gpu::createTritonTensorMemoryAllocationPass());
  pm.addPass(mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass());

  {
    mlir::triton::nvidia_gpu::TritonGPUProxyFenceInsertionOptions options;
    options.computeCapability = capability;
    pm.addPass(mlir::triton::nvidia_gpu::createTritonGPUProxyFenceInsertion(options));
  }
  
  pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(capability, ptxVersion));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::triton::createConvertNVGPUToLLVM());
  pm.addPass(mlir::triton::createConvertWarpSpecializeToLLVM());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addPass(mlir::createConvertNVVMToLLVMPass());

  // Apply the pass
  if (failed(pm.run(module.get()))) {
    llvm::errs() << "Pass execution failed\n";
    return 1;
  }

  // Print the resulting module
  llvm::outs() << "Lowered MLIR:\n";
  module->print(llvm::outs());

  // LLVM-IR (MLIR) -> LLVM-IR (LLVM)
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  });

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmMod = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
  if (!llvmMod) {
    llvm::errs() << "failed to translate module to LLVM IR\n";
    std::terminate();
  }

  std::string arch = "sm_";
  arch.append(std::to_string(capability));

  std::string features = "+ptx";
  features.append(std::to_string(capability));

  std::string triple = "nvptx64-nvidia-cuda";

  auto options = llvm::cl::getRegisteredOptions();
  const char *flag = "nvptx-short-ptr";
  auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
  assert(shortPtr);
  shortPtr->setValue(true);

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    llvm::errs() << "target lookup error: " + error << "\n";
    std::terminate();
  }

  llvm::TargetOptions opt;
  // Target machine is only used to create the data layout.
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      llvm::Triple(triple), arch, features, opt, llvm::Reloc::PIC_,
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

  std::string libDevice = "third_party/nvidia/backend/lib/libdevice.10.bc";
  std::vector<std::string> paths{libDevice};
  linkExternLibs(llvmMod.get(), paths);

  optimizeLLVMModule(llvmMod.get(), llvm::OptimizationLevel::O3);

  llvm::outs() << "LLVMIR:\n";
  llvm::outs() << *llvmMod << "\n";

  return 0;
}