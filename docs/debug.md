# Debug
```C++
// main.cc
#include "llvm/Support/Debug.h"

LLVM_DEBUG(llvm::dbgs() << "Start initializing " << getDialectNamespace() << "\n");
```

Run:
```Bash
./build/main -debug

# Somtimes, you may need the following flag to debug mlir.
# However, something must be configured to make it available.
--mlir-print-debuginfo
```

## error: expected 2 dynamic size values
llvm-project/mlir/lib/Interfaces/ViewLikeInterface.cpp

## Print pass name
llvm-project/mlir/lib/Pass/Pass.cpp
```C++
//===----------------------------------------------------------------------===//
// OpToOpPassAdaptor
//===----------------------------------------------------------------------===//

LogicalResult OpToOpPassAdaptor::run(Pass *pass, Operation *op,
                                     AnalysisManager am, bool verifyPasses,
                                     unsigned parentInitGeneration) {
  llvm::outs() << "Running pass: " << pass->getName() << "\n";
```