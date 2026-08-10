# How to create custom MLIR passes
Steps are as follows:

## 1. Table Definition
Example: [../examples/lower_mhlo_broadcast/passes.td](../examples/lower_mhlo_broadcast/passes.td)

## 2. Table Gen
Example: [../examples/lower_mhlo_broadcast/CMakeLists.txt](../examples/lower_mhlo_broadcast/CMakeLists.txt)
```cmake
#============================================================================
# Custom passes
#============================================================================
set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CustomPassesIncGen)
```
When `cmake --build build/` is executed, `build/passes.h.inc` will generated.

## 3. Create header passes.h
Define a function for creating the pass.

Example: [../examples/lower_mhlo_broadcast/passes.h](../examples/lower_mhlo_broadcast/passes.h)

## 4. Create source passes.cc
Example: [../examples/lower_mhlo_broadcast/passes.cc](../examples/lower_mhlo_broadcast/passes.cc)

Function and Base class definitions:
```cpp
namespace mlir {

#define GEN_PASS_DEF_CUSTOMCOMPUTEOPANDFUNCBUFFERIZEPASS
#include "passes.h.inc"

} // namespace mlir
```

Actual pass implementation via CRTP:
```cpp
using namespace mlir;

namespace {

struct CustomComputeOpAndFuncBufferizePass : public impl::CustomComputeOpAndFuncBufferizePassBase<CustomComputeOpAndFuncBufferizePass> {

void runOnOperation() override {
  // Implement the pass's logic here
}

};

} // anonymous namespace
```