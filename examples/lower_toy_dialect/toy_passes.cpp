#include "mlir/IR/Verifier.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


#include "toy_dialect.h"

using namespace mlir;

namespace mlir::toy {

#define GEN_PASS_DEF_CONVERTTOYTOARITHPASS
#define GEN_PASS_DEF_CONVERTTENSORTOMEMREFPASS
#include "toy_passes.h.inc"

} // namespace mlir::toy

namespace {

using namespace mlir::toy;

// Conversion pattern for toy.add to arith.addf
struct ToyAddLowering : public OpConversionPattern<toy::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(toy::AddOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // Check that the input and output types are tensor<f64>
    auto inputType = dyn_cast<TensorType>(op.getLhs().getType());
    if (!inputType || !isa<Float64Type>(inputType.getElementType()))
      return failure();

    // Create arith.addf operation
    auto resultType = op.getResult().getType();
    Value addOp = rewriter.create<arith::AddFOp>(op.getLoc(), adaptor.getLhs(), adaptor.getRhs());

    // Replace the toy.add operation with the new arith.addf
    rewriter.replaceOp(op, addOp);
    return success();
  }
};

// Pass to lower toy dialect to arith dialect
struct ConvertToyToArithPass : public toy::impl::ConvertToyToArithPassBase<ConvertToyToArithPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    // TypeConverter

    // Define the conversion target (arith dialect is legal)
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<toy::ToyDialect>();

    // Define the conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<ToyAddLowering>(&getContext());

    // std::unique_ptr<DenseSet<Operation *>> legalizableOps(new DenseSet<Operation *>());
    // ConversionConfig config;
    // config.legalizableOps = legalizableOps.get();
    // if (failed(applyAnalysisConversion(getOperation(), target, std::move(patterns), config))) {
    //   signalPassFailure();
    // }

    // llvm::outs() << "Legalizable ops:\n";
    // for (auto op : *config.legalizableOps) {
    //   llvm::outs() << *op << "\n";
    // }
    // llvm::outs() << "\n";

    // patterns.clear();
    // patterns.add<ToyAddLowering>(&getContext());

    // Apply the conversion
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};


// Step 1: Define the TypeConverter
struct TensorToMemRefConverter : public TypeConverter {
  TensorToMemRefConverter() {
    // Map tensor to memref
    addConversion([](TensorType tensor) -> MemRefType {
      auto rankedTensor = dyn_cast<RankedTensorType>(tensor);
      return MemRefType::get(rankedTensor.getShape(), rankedTensor.getElementType());
    });

    // Allow memref types to pass through unchanged
    addConversion([](MemRefType memref) { return memref; });

    // Register source materialization: memref<?xf64> -> tensor<?xf64>
    addSourceMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange convertedValues, mlir::Location loc) -> mlir::Value {
          assert(convertedValues.size() == 1 && "convertedValues must have size = 1");
          auto srcValue = builder.create<UnrealizedConversionCastOp>(loc, resultType, convertedValues[0]);
          // return mlir::Value();
          return srcValue.getOutputs()[0];
        });

    // Register target materialization: tensor<?xf64> -> memref<?xf64>
    addTargetMaterialization(
        [](mlir::OpBuilder &builder, mlir::TypeRange resultTypes,
           mlir::ValueRange srcValues, mlir::Location loc) -> SmallVector<Value> {

          SmallVector<Value> tgtValues;
          for (const auto& [t, v] : llvm::zip(resultTypes, srcValues)) {
            auto ret = builder.create<UnrealizedConversionCastOp>(loc, t, v);
            llvm::outs() << "\nMaterialized " << ret << "\n";
            tgtValues.push_back(ret.getOutputs()[0]);
          }

          return tgtValues;
        });
  }
};

// Step 2: Function Conversion Pattern
struct FuncOpConverter : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::FuncOp oldFunc, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    llvm::outs() << "\nBegin FuncOpConverter:\n" << *oldFunc->getParentOp() << "\n";

    //===============================================================================================
    // 1. Create newFunc
    //===============================================================================================
    auto typeConverter = getTypeConverter();
    auto oldFuncType = oldFunc.getFunctionType();

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

    auto newFuncType = FunctionType::get(getContext(), newInputTypes, newResultTypes);
    auto newFunc = rewriter.create<func::FuncOp>(oldFunc.getLoc(), oldFunc.getName(), newFuncType);

    // Copy attrs except the type
    for (auto attr : oldFunc->getAttrs()) {
      if (attr.getName() != oldFunc.getFunctionTypeAttrName()) {
        newFunc->setAttr(attr.getName(), attr.getValue());
      }
    }

    newFunc.setVisibility(oldFunc.getVisibility());

    // Move the body.
    rewriter.inlineRegionBefore(oldFunc.getBody(), newFunc.getBody(), newFunc.end());

    //===============================================================================================
    // 2. Convert the entry block's arguments
    //===============================================================================================
    // Get the old entry block (now inside newFunc).
    Block& oldEntry = newFunc.getBody().front();

    // 1) Create a fresh entry block with converted argument types.
    SmallVector<Location> argLocs;
    argLocs.reserve(oldEntry.getNumArguments());
    for (BlockArgument a : oldEntry.getArguments()) {
      argLocs.push_back(a.getLoc());
    }

    SmallVector<Type> newArgTypes(newFuncType.getInputs().begin(), newFuncType.getInputs().end());

    // Insert the new block before the old one.
    Block *newEntry = rewriter.createBlock(&newFunc.getBody(), newFunc.getBody().begin(), newArgTypes, argLocs);

    // 2) Materialize conversions from newEntry args to the types expected by oldEntry.
    // We need a mapping: newArgs -> bridgeArgs (with old types) -> uses
    SmallVector<Value> bridgeArgs;
    bridgeArgs.reserve(oldEntry.getNumArguments());
    rewriter.setInsertionPoint(newEntry, newEntry->begin());
    for (auto pair : llvm::zip(oldEntry.getArguments(), newEntry->getArguments())) {
      Value oldArg = std::get<0>(pair);
      Value newArg = std::get<1>(pair);
      Type oldTy = oldArg.getType();
      Type newTy = newArg.getType();
      Value bridge;
      if (oldTy != newTy) {
        // Ask the TypeConverter to materialize a conversion at the block boundary.
        bridge = typeConverter->materializeSourceConversion(rewriter, oldArg.getLoc(), oldTy, newArg);
        if (!bridge) {
          llvm::outs() << "\nCurrent MLIR module state:\n";
          llvm::outs() << *oldFunc->getParentOp() << "\n\n";
          llvm::errs() << "failed to materialize conversion from " << newTy << " to " << oldTy;
          return failure();
        }
      }

      bridgeArgs.push_back(bridge);
    }

    llvm::outs() << "\nBefore update block arguments:\n";
    llvm::outs() << *oldFunc->getParentOp() << "\n\n";

    // 3) Replace all uses of old entry arguments inside the oldEntry with the bridged values.
    for (auto pair : llvm::zip(oldEntry.getArguments(), bridgeArgs)) {
      rewriter.replaceAllUsesWith(std::get<0>(pair), std::get<1>(pair));
    }

    llvm::outs() << "\nAfter update block arguments:\n";
    llvm::outs() << *oldFunc->getParentOp() << "\n\n";

    // 4) Move the operations from oldEntry to newEntry and erase the old block.
    rewriter.inlineBlockBefore(&oldEntry, newEntry, newEntry->end(), newEntry->getArguments());

    llvm::outs() << "\nAfter move the entry block:\n";
    llvm::outs() << *oldFunc->getParentOp() << "\n\n";

    //===============================================================================================
    // 3. Replace oldFunc with newFunc
    //===============================================================================================
    rewriter.replaceOp(oldFunc, newFunc);

    llvm::outs() << "\nAfter FuncOpConverter:\n" << *newFunc->getParentOp() << "\n\n";

    return success();
  }
};

struct FuncReturnOpConverter : public OpConversionPattern<func::ReturnOp> {
  // using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
  FuncReturnOpConverter(MLIRContext *context, PatternBenefit benefit = 99)
      : OpConversionPattern(context, benefit) {}
  FuncReturnOpConverter(const TypeConverter &typeConverter, MLIRContext *context, PatternBenefit benefit = 99)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult matchAndRewrite(
      func::ReturnOp returnOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "\nBegin FuncReturnOpConverter:\n";
    llvm::outs() << *returnOp->getParentOp() << "\n\n";

    auto operands = adaptor.getOperands();
    auto newReturnOp = rewriter.create<func::ReturnOp>(returnOp.getLoc(), operands);
    rewriter.replaceOp(returnOp, newReturnOp);

    llvm::outs() << "\nAfter FuncReturnOpConverter:\n";
    llvm::outs() << *newReturnOp->getParentOp() << "\n\n";
    return success();
  }
};

// Step 3: AddFOpConverter Conversion Pattern
struct AddFOpConverter : public OpConversionPattern<arith::AddFOp> {
  using OpConversionPattern<arith::AddFOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::AddFOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto parentOp = op->getParentOp();
    llvm::outs() << "\nBegin AddFOpConverter:\n" << *parentOp << "\n";

    auto xMemref = adaptor.getOperands()[0];
    auto yMemref = adaptor.getOperands()[1];

    auto outputMemrefType = cast<MemRefType>(getTypeConverter()->convertType(op.getResult().getType()));
    int64_t rank = outputMemrefType.getRank();

    // Allocate output memref with dynamic sizes.
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < rank; ++i) {
      if (outputMemrefType.isDynamicDim(i)) {
        Value dimSize = rewriter.create<memref::DimOp>(op.getLoc(), xMemref, i);
        dynSizes.push_back(dimSize);
      }
    }
    Value outputMemref = rewriter.create<memref::AllocaOp>(op.getLoc(), outputMemrefType, dynSizes);

    // Indexing maps: identity for x and output, empty or identity for y.
    SmallVector<AffineExpr> exprs;
    for (int64_t i = 0; i < rank; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    auto xIndexMap = AffineMap::get(rank, 0, exprs, rewriter.getContext());
    auto yIndexMap = xIndexMap;
    auto outputIndexMap = xIndexMap;
    SmallVector<AffineMap> indexingMaps = {xIndexMap, yIndexMap, outputIndexMap};

    // Set iterator types: all parallel for element-wise operation.
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);

    // Create linalg.generic operation with memref semantics.
    auto linalgOp = rewriter.create<linalg::GenericOp>(
      op.getLoc(),
      /*resultTypes=*/TypeRange{}, // No tensor results; output is written to memref
      /*inputs=*/ValueRange{xMemref, yMemref},
      /*outputs=*/ValueRange{outputMemref},
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange args) {
        Value xVal = args[0];
        Value yVal = args[1];
        Value result = nestedBuilder.create<arith::AddFOp>(loc, xVal, yVal);
        nestedBuilder.create<linalg::YieldOp>(loc, result);
      }
    );

    rewriter.replaceOp(op, outputMemref);

    llvm::outs() << "\nAfter AddFOpConverter:\n" << *parentOp << "\n";

    return success();
  }
};

// Step 4: Define the Pass
struct ConvertTensorToMemRefPass : public toy::impl::ConvertTensorToMemRefPassBase<ConvertTensorToMemRefPass> {
  static bool checkOpLegality(Operation* op) {
    if (op->getNumOperands() > 0) {
      for (const auto& opr : op->getOperands()) {
        if (isa<TensorType>(opr.getType())) {
          llvm::outs() << "\nOperation (" << op << ") " << op->getName().getStringRef() << " is not legal ❌\n";
          return false;
        }
      }
    }
    llvm::outs() << "\nOperation (" << op << ") " << op->getName().getStringRef() << " is legal ✅\n";
    return true;
  }

  void runOnOperation() override {
    auto module = getOperation();
    TensorToMemRefConverter typeConverter;

    // Define conversion target
    ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();

    for (auto& op : module.getBody()->getOperations()) {
      if (auto funcOp = dyn_cast<func::FuncOp>(&op)) {
        for (auto& innerOp : funcOp.getBody().front().getOperations()) {
          target.addDynamicallyLegalOp(innerOp.getName(), checkOpLegality);
        }
      }
    }

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto isLegal = typeConverter.isSignatureLegal(op.getFunctionType());
      if (!isLegal) {
        llvm::outs() << "\nFuncOp " << op.getOperation() << " is not legal ❌\n";
        return false;
      }

      llvm::outs() << "\nFuncOp is legal ✅\n";
      return true;
    });

    // Populate conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<FuncOpConverter, FuncReturnOpConverter>(typeConverter, &getContext());
    patterns.add<AddFOpConverter>(typeConverter, &getContext());

    // Apply partial conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}