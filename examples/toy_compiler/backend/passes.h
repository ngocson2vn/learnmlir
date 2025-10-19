#pragma once

#include <memory>
#include "mlir/Pass/Pass.h"


namespace mlir {

class ModuleOp;

namespace func {
class FuncOp;
}

namespace toy {

// Creates a TileLoopsPass with tiles sizes provided through `tile_sizes`
// and unroll factors provided through `unroll_factors`.
std::unique_ptr<OperationPass<func::FuncOp>> createTileLoopsPass(
    ArrayRef<int64_t> tileSizes = {}, ArrayRef<int64_t> unrollFactors = {});


std::unique_ptr<mlir::OperationPass<ModuleOp>> createGpuModuleToCubinPass();

}  // namespace toy
}  // namespace mlir
