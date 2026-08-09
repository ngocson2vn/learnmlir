#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mhlo/IR/hlo_ops.h"

namespace mlir {

#define GEN_PASS_DEF_CUSTOMCOMPUTEOPANDFUNCBUFFERIZEPASS
#include "passes.h.inc"

} // namespace mlir

using namespace mlir;

namespace {

struct CustomComputeOpAndFuncBufferizePass : public impl::CustomComputeOpAndFuncBufferizePassBase<CustomComputeOpAndFuncBufferizePass> {
  bufferization::BufferizationOptions getPartialBufferizationOptions() {
    bufferization::BufferizationOptions options;
    options.allowUnknownOps = true;
    options.unknownTypeConverterFn = [](TensorType type, Attribute memorySpace,
                                        const bufferization::BufferizationOptions &options) {
      return bufferization::getMemRefTypeWithStaticIdentityLayout(type, memorySpace);
    };

    return options;
  }

  void runOnOperation() override {
    // Bufferize ops using BufferizableOpInterface.
    bufferization::BufferizationOptions options = getPartialBufferizationOptions();
    options.opFilter.allowDialect<bufferization::BufferizationDialect,
                                  linalg::LinalgDialect, mhlo::MhloDialect,
                                  shape::ShapeDialect, tensor::TensorDialect
                                  >();
    bufferization::BufferizationState state;
    if (failed(bufferization::bufferizeOp(getOperation(), options, state))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace