#include "toy_dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

// Conversion pattern for toy.add to arith.addf
struct ToyAddLowering : public OpConversionPattern<toy::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    toy::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Check that the input and output types are tensor<f64>
    auto inputType = dyn_cast<TensorType>(op.getLhs().getType());
    if (!inputType || !isa<Float64Type>(inputType.getElementType()))
      return failure();

    // Create arith.addf operation
    auto resultType = op.getResult().getType();
    Value addOp = rewriter.create<arith::AddFOp>(
        op.getLoc(), adaptor.getLhs(), adaptor.getRhs());

    // Replace the toy.add operation with the new arith.addf
    rewriter.replaceOp(op, addOp);
    return success();
  }
};

// Pass to lower toy dialect to arith dialect
struct ConvertToyToArith
    : public PassWrapper<ConvertToyToArith, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    // Define the conversion target (arith dialect is legal)
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<toy::ToyDialect>();

    // Define the conversion patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<ToyAddLowering>(&getContext());

    // Apply the conversion
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// Register the pass
namespace {
struct ConvertToyToArithPass
    : public PassRegistration<ConvertToyToArith> {};
} // namespace
