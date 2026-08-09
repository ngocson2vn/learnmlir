#pragma once
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "passes.h.inc"

} // namespace mlir
