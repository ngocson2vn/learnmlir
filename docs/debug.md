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