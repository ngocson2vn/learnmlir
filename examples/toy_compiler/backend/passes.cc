#include <fstream>
#include <filesystem>

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Passes/OptimizationLevel.h"

#include "passes.h"
#include "utils.h"
#include "cuda_utils.h"
#include "llvm_utils.h"
#include "nvidia_backend.h"

using namespace mlir;

namespace mlir::toy {

#define GEN_PASS_DECL_GPUMODULETOCUBINPASS
#define GEN_PASS_DEF_GPUMODULETOCUBINPASS
#include "backend/passes.h.inc"

} // namespace mlir::toy

namespace {

struct GPUModuleOpPattern : public mlir::OpConversionPattern<gpu::GPUModuleOp> {
  GPUModuleOpPattern(MLIRContext *context, int cc)
    : OpConversionPattern(context), capability_(cc) {
  }

  LogicalResult matchAndRewrite(gpu::GPUModuleOp gpuModule, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::LLVMContext llvmContext;
    auto llvmMod = mlir::translateModuleToLLVMIR(gpuModule, llvmContext);
    if (!llvmMod) {
      llvm::errs() << "Failed to translate module to LLVM IR\n";
      return failure();
    }

    // LLVMIR to PTX
    if (failed(nvidia::linkLibdevice(llvmMod.get(), capability_))) {
      return failure();
    }

    llvm::optimizeLLVMModule(llvmMod.get(), llvm::OptimizationLevel::O3, nvidia::kTriple);

    std::string arch = std::string("sm_").append(std::to_string(capability_));
    auto ptxVersion = cuda::getSupportedPtxVersion();
    std::string features = std::string("+ptx").append(ptxVersion);
    auto ptx = nvidia::translateLLVMIRToPTX(llvmMod.get(), arch, features);
    if (ptx.empty()) {
      llvm::errs() << "Failed to translate LLVMIR to PTX\n";
      return failure();
    }

    std::filesystem::path currentPath = std::filesystem::current_path();
    std::string outputPtxFile = currentPath.string() + "/output.ptx";
    bool ok = ::toy::utils::writeFile(ptx, outputPtxFile);
    if (!ok) {
      llvm::errs() << "Failed to save PTX to outputPtxFile\n";
      return failure();
    }

    // PTX to CUBIN
    std::string outputCubinFile = currentPath.string() + "/output.cubin";
    auto cubinStr = nvidia::translatePTXtoCUBIN(outputPtxFile, arch, outputCubinFile);
    if (cubinStr.empty()) {
      llvm::errs() << "Failed to translate PTX to CUBIN\n";
      return failure();
    }

    auto context = gpuModule.getContext();
    auto symName = gpuModule.getSymName();

    rewriter.startOpModification(gpuModule);
    gpuModule.setSymName(symName.str() + "_old");

    auto modOp = gpuModule->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPoint(&modOp.front());
    mlir::SmallVector<mlir::Attribute> objects;
    auto nvvmTarget = mlir::NVVM::NVVMTargetAttr::get(context, 3, nvidia::kTriple, arch, features);
    auto offloadingHandler = mlir::gpu::SelectObjectAttr::get(context, nvvmTarget);
    auto gpuObj = mlir::gpu::ObjectAttr::get(nvvmTarget, mlir::gpu::CompilationTarget::Binary, rewriter.getStringAttr(cubinStr));
    objects.push_back(gpuObj);
    auto gpuBinOp = rewriter.create<mlir::gpu::BinaryOp>(gpuModule.getLoc(), symName, offloadingHandler, objects);

    rewriter.finalizeOpModification(gpuModule);
    rewriter.eraseOp(gpuModule);

    return success();
  }

 private:
  int capability_;
};

class GpuModuleToCubinPass : public mlir::toy::impl::GpuModuleToCubinPassBase<GpuModuleToCubinPass> {
  void runOnOperation() override {
    llvm::initTargets();

    auto context = &getContext();
    auto module = getOperation();

    ConversionTarget target(*context);
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<NVVM::NVVMDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<gpu::GPUModuleOp>();

    RewritePatternSet patterns(context);
    patterns.add<GPUModuleOpPattern>(context, capability_);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> 
mlir::toy::createGpuModuleToCubinPass() {
  return std::make_unique<GpuModuleToCubinPass>();
}
