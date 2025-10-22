#pragma once

#include <memory>
#include "mlir/Pass/Pass.h"


namespace mlir {

class ModuleOp;

namespace toy {

std::unique_ptr<mlir::OperationPass<ModuleOp>> createLowerMemRefToLLVMPass();

}  // namespace toy
}  // namespace mlir
