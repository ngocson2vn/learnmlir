#include <fstream>
#include <filesystem>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "backend-passes"

#include "passes.h"
#include "utils.h"
#include "cuda_utils.h"

using namespace mlir;

namespace mlir::toy {

#define GEN_PASS_DEF_LOWERMEMREFTOLLVMPASS
#include "backend/passes.h.inc"

} // namespace mlir::toy

namespace {

struct LowerMemRefToLLVMFuncOpPattern : public mlir::OpConversionPattern<func::FuncOp> {  
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::FuncOp oldFunc, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
    auto typeConverter = this->getTypeConverter();
    //===============================================================================================
    // 1. Create newFunc
    //===============================================================================================
    auto oldFuncType = cast<FunctionType>(oldFunc.getFunctionType());

    // Convert input types
    SmallVector<Type> newInputTypes;
    if (failed(typeConverter->convertTypes(oldFuncType.getInputs(), newInputTypes))) {
      return failure();
    }

    // Convert result types
    SmallVector<Type> newResultTypes;
    if (failed(typeConverter->convertTypes(oldFuncType.getResults(), newResultTypes))) {
      return failure();
    }

    if (newResultTypes.size() > 1) {
      llvm::errs() << "The function " << oldFunc.getName() << " returns more than 1 value\n";
      return failure();
    }

    auto resType = newResultTypes.size() == 1 ? newResultTypes[0] : LLVM::LLVMVoidType::get(oldFunc.getContext());

    auto newFuncType = LLVM::LLVMFunctionType::get(resType, newInputTypes);
    auto newFunc = rewriter.create<LLVM::LLVMFuncOp>(oldFunc.getLoc(), oldFunc.getName(), newFuncType);

    // Copy attrs except the type
    for (auto attr : oldFunc->getAttrs()) {
      if (attr.getName() != newFunc.getFunctionTypeAttrName()) {
        newFunc->setAttr(attr.getName(), attr.getValue());
      }
    }

    newFunc.setVisibility(oldFunc.getVisibility());

    // Move the body.
    rewriter.inlineRegionBefore(oldFunc.getFunctionBody(), newFunc.getBody(), newFunc.end());
    
    Block& entryBlock = newFunc.front();
    auto sig = typeConverter->convertBlockSignature(&entryBlock);
    if (!sig.has_value()) {
      llvm::errs() << "Failed to convert entry block signature\n";
      return failure();
    }

    rewriter.applySignatureConversion(&entryBlock, sig.value(), typeConverter);

    //===============================================================================================
    // 3. Replace oldFunc with newFunc
    //===============================================================================================
    rewriter.replaceOp(oldFunc, newFunc);
    LLVM_DEBUG(llvm::dbgs() << "\nAfter FuncOpConverter:\n" << *newFunc->getParentOp() << "\n\n");

    return success();
  }
};

struct LowerMemRefToLLVMReturnOpPattern : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::ReturnOp oldReturnOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto newReturnOp = rewriter.create<LLVM::ReturnOp>(oldReturnOp.getLoc(), oldReturnOp.getOperands());
    rewriter.replaceOp(oldReturnOp, newReturnOp);

    return success();
  }
};


struct LowerMemRefToLLVMLoadOpPattern : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::LoadOp loadOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto elemType = loadOp.getMemRefType().getElementType();
    auto resType = LLVM::LLVMPointerType::get(getContext());
    auto operands = adaptor.getOperands();

    // Type resultType, Type elementType, Value basePtr, ValueRange indices
    auto loadPtr = rewriter.create<LLVM::GEPOp>(loadOp.getLoc(), resType, elemType, operands[0], operands[1]);
    auto newOp = rewriter.create<LLVM::LoadOp>(loadOp.getLoc(), elemType, loadPtr);
    rewriter.replaceOp(loadOp, newOp);

    return success();
  }
};

struct LowerMemRefToLLVMStoreOpPattern : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::StoreOp storeOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto elemType = storeOp.getMemRefType().getElementType();
    auto resType = LLVM::LLVMPointerType::get(getContext());
    auto operands = adaptor.getOperands();

    // Type resultType, Type elementType, Value basePtr, ValueRange indices
    auto storePtr = rewriter.create<LLVM::GEPOp>(storeOp.getLoc(), resType, elemType, operands[1], operands[2]);
    auto newOp = rewriter.create<LLVM::StoreOp>(storeOp.getLoc(), operands[0], storePtr);
    rewriter.replaceOp(storeOp, newOp);

    return success();
  }
};

struct LowerMemRefToLLVMTypeConverter : public TypeConverter {
  LowerMemRefToLLVMTypeConverter() {
    addConversion([](Type srcType) -> Type {
      if (auto memrefType = dyn_cast<MemRefType>(srcType)) {
        return LLVM::LLVMPointerType::get(srcType.getContext());
      }

      if (auto indexType = dyn_cast<IndexType>(srcType)) {
        return IntegerType::get(srcType.getContext(), 64);
      }

      return srcType;
    });

    // Register source materialization: memref<?xf64> -> tensor<?xf64>
    addSourceMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange convertedValues, mlir::Location loc) -> mlir::Value {
          assert(convertedValues.size() == 1 && "convertedValues must have size = 1");
          auto srcValue = builder.create<arith::IndexCastOp>(loc, resultType, convertedValues[0]);
          return srcValue;
        });

    // Register target materialization: tensor<?xf64> -> memref<?xf64>
    addTargetMaterialization(
        [](mlir::OpBuilder &builder, mlir::TypeRange resultTypes,
           mlir::ValueRange srcValues, mlir::Location loc) -> SmallVector<Value> {

          SmallVector<Value> tgtValues;
          for (const auto& [t, v] : llvm::zip(resultTypes, srcValues)) {
            auto ret = builder.create<arith::IndexCastOp>(loc, t, v);
            tgtValues.push_back(ret);
          }

          return tgtValues;
        });
  }
};

class LowerMemRefToLLVMPass : public mlir::toy::impl::LowerMemRefToLLVMPassBase<LowerMemRefToLLVMPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Define type converter
    LowerMemRefToLLVMTypeConverter typeConverter;

    // Define conversion target
    ConversionTarget target(getContext());

    //===========================================================================
    // 1. Convert function signature
    //===========================================================================
    target.addLegalDialect<LLVM::LLVMDialect>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto isLegal = typeConverter.isSignatureLegal(op.getFunctionType());
      if (!isLegal) {
        LLVM_DEBUG(llvm::dbgs() << "\nFuncOp " << op.getOperation() << " is not legal ❌\n");
        return false;
      }

      LLVM_DEBUG(llvm::dbgs() << "\nFuncOp is legal ✅\n");
      return true;
    });

    // Populate conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<
      LowerMemRefToLLVMFuncOpPattern,
      LowerMemRefToLLVMLoadOpPattern,
      LowerMemRefToLLVMStoreOpPattern,
      LowerMemRefToLLVMReturnOpPattern
    >(typeConverter, &getContext());

    // Apply partial convertion
    ConversionConfig config;
    config.buildMaterializations = true;
    if (failed(applyPartialConversion(module, target, std::move(patterns), config))) {
      signalPassFailure();
      LLVM_DEBUG(llvm::dbgs() << "\nRestored module:\n");
      LLVM_DEBUG(llvm::dbgs() << module << "\n");
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<ModuleOp>> 
mlir::toy::createLowerMemRefToLLVMPass() {
  return std::make_unique<LowerMemRefToLLVMPass>();
}
