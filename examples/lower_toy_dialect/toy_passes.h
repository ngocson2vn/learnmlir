#pragma once
#include "mlir/Pass/Pass.h"

namespace mlir::toy {

#define GEN_PASS_DECL
#include "toy_passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "toy_passes.h.inc"

} // namespace mlir::toy
