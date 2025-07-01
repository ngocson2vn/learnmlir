#ifndef DISC_TRANSFORMS_PASSES_H_
#define DISC_TRANSFORMS_PASSES_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


namespace mlir {
namespace disc_ral {

void registerAllDiscPasses();

// Lowers the roots of lmhlo.fusion to parallel loops
std::unique_ptr<OperationPass<func::FuncOp>>
createDiscLhloLegalizeRootsToParallelLoopsPass(int sm_count = -1,
                                               int cc_major = 8,
                                               int cc_minor = 0);

}  // namespace disc_ral
}  // namespace mlir

#endif  // DISC_TRANSFORMS_PASSES_H_
