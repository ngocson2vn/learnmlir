# MLIR Language Reference
https://mlir.llvm.org/docs/LangRef/

MLIR is designed to be used in three different forms: a human-readable textual form suitable for debugging, an in-memory form suitable for programmatic transformations and analysis, and a compact serialized form suitable for storage and transport. The different forms all describe the same semantic content. This document describes the human-readable textual form.

## High-Level Structure
nodes == Operations
edges == Values

## Notation
This document describes the grammar using Extended Backus-Naur Form (EBNF).

# Notes
Any Language -> Dialect + MLIR -> MLIR IR -> Analyze -> Transform/Optimize -> MLIR IR

## Dump an Operation
```C++
auto& front = islandOp.front();
front.print(llvm::outs());
```

## SSA
Single Static Assignment (SSA) is a property of an intermediate representation (IR), which requires that each variable is assigned exactly once and every variable is defined before it is used. SSA simplifies and improves the efficiency of various compiler optimizations because it makes the data flow properties of a program explicit.

Simple Example
Consider the following code in a non-SSA form:
```C++
int a = 1;
int b = 2;
int c = a + b;
a = 3;
c = a + b;
```
In SSA form, each assignment to a variable results in a new version of that variable:
```C++
int a1 = 1;
int b1 = 2;
int c1 = a1 + b1;
int a2 = 3;
int c2 = a2 + b1;
```
Here, a1 and a2 are two different versions of the variable a. This makes it clear which value is used in each computation.

## getUsers()
The "users" of an SSA value are instances of Operation, while the "uses" refer to the operands of these operations. For example considering test.op(%0, %0) : ..., when iterating on the “uses” of %0 you would see two instances of OpOperand (one for each use in test.op), whereas iterating on the “users” of %0 would yield directly two Operation * corresponding to test.op. Note that you see test.op twice as it is twice a user of %0, it’s up to the call site to use a set to unique these if needed. The tutorial on use-def chains may help understand the details as well.

## Get op name
-exec p user->getName().getStringRef()

## Operation
llvm-project/mlir/include/mlir/IR/Operation.h
```C++
namespace mlir {

/// Operation is a basic unit of execution within MLIR. Operations can
/// be nested within `Region`s held by other operations effectively forming a
/// tree. Child operations are organized into operation blocks represented by a
/// 'Block' class.
class alignas(8) Operation final
    : public llvm::ilist_node_with_parent<Operation, Block>,
      private llvm::TrailingObjects<Operation, detail::OperandStorage,
                                    BlockOperand, Region, OpOperand> {
public:
  /// The name of an operation is the key identifier for it.
  OperationName getName() { return name; }

private:
  /// This holds the name of the operation.
  OperationName name;
}

} // namespace mlir
```

llvm-project/mlir/include/mlir/IR/OperationSupport.h
```C++
namespace mlir {

class OperationName {
public:
  OperationName(StringRef name, MLIRContext *context);

  /// Return the name of this operation. This always succeeds.
  StringRef getStringRef() const { return getIdentifier(); }

  /// Return the name of this operation as a StringAttr.
  StringAttr getIdentifier() const { return impl->name; }

protected:
  /// This class represents a type erased version of an operation. It contains
  /// all of the components necessary for opaquely interacting with an
  /// operation. If the operation is not registered, some of these components
  /// may not be populated.
  struct Impl {
    Impl(StringAttr name)
        : name(name), dialect(nullptr), interfaceMap(llvm::None) {}

    /// The name of the operation.
    StringAttr name;
  };

protected:
  OperationName(Impl *impl) : impl(impl) {}

  /// The internal implementation of the operation name.
  Impl *impl;
}

} // namespace mlir
```

# Pass Manager
There are two main classes related to pass management, the `PassManager` and the `OpPassManager`. The `PassManager` class acts as the top-level entry point, and contains various configurations used for the entire pass pipeline. The `OpPassManager` class is used to schedule passes to run at a specific level of nesting. The top-level `PassManager` also functions as an `OpPassManager`.  
```C++
mlir::OpTrait::SingleBlock<mlir::tf_executor::IslandOp>::front<mlir::tf_executor::IslandOp>(mlir::OpTrait::SingleBlock<mlir::tf_executor::IslandOp> * const this) (/data00/home/son.nguyen/workspace/auto_fusion_dev/erdos/operators/auto_fusion/build/4e26b3ed4690a11c0fad61432f5a413b/external/llvm-raw/mlir/include/mlir/IR/OpDefinition.h:869)
mlir::detail::(anonymous namespace)::SimpleFusion::clusterGraph(const mlir::detail::(anonymous namespace)::SimpleFusion * const this, mlir::tf_executor::GraphOp graph, const mlir::detail::OpsPredecessors * opsPredecessors) (/data00/home/son.nguyen/workspace/auto_fusion_dev/erdos/operators/auto_fusion/lib/Transforms/cluster_algo.cc:2160)
(anonymous namespace)::FuseCwiseOps::runOnOperation((anonymous namespace)::FuseCwiseOps * const this) (/data00/home/son.nguyen/workspace/auto_fusion_dev/erdos/operators/auto_fusion/lib/Transforms/fuse_cwise_ops.cc:1396)
mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (Unknown Source:0)
mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (Unknown Source:0)
mlir::detail::OpToOpPassAdaptor::runOnOperationAsyncImpl(bool) (Unknown Source:0)
mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) (Unknown Source:0)
mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) (Unknown Source:0)
mlir::PassManager::run(mlir::Operation*) (Unknown Source:0)
performActions(llvm::raw_ostream & os, bool verifyDiagnostics, bool verifyPasses, llvm::SourceMgr & sourceMgr, mlir::MLIRContext * context, mlir::PassPipelineFn passManagerSetupFn, bool emitBytecode, bool implicitModule) (/data00/home/son.nguyen/workspace/auto_fusion_dev/erdos/operators/auto_fusion/build/4e26b3ed4690a11c0fad61432f5a413b/external/llvm-raw/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:91)
processBuffer(llvm::raw_ostream & os, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> > ownedBuffer, bool verifyDiagnostics, bool verifyPasses, bool allowUnregisteredDialects, bool preloadDialectsInContext, bool emitBytecode, bool implicitModule, mlir::PassPipelineFn passManagerSetupFn, mlir::DialectRegistry & registry, llvm::ThreadPool * threadPool) (/data00/home/son.nguyen/workspace/auto_fusion_dev/erdos/operators/auto_fusion/build/4e26b3ed4690a11c0fad61432f5a413b/external/llvm-raw/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:141)
mlir::<lambda(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>::operator()(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream &) const(const mlir::<lambda(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)> * const __closure, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> > chunkBuffer, llvm::raw_ostream & os) (/data00/home/son.nguyen/workspace/auto_fusion_dev/erdos/operators/auto_fusion/build/4e26b3ed4690a11c0fad61432f5a413b/external/llvm-raw/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:184)
llvm::function_ref<mlir::LogicalResult(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer>, mlir::PassPipelineFn, mlir::DialectRegistry&, bool, bool, bool, bool, bool, bool, bool)::<lambda(std::unique_ptr<llvm::MemoryBuffer>, llvm::raw_ostream&)> >(intptr_t, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream &)(intptr_t callable,  params#0,  params#1) (/data00/home/son.nguyen/workspace/auto_fusion_dev/erdos/operators/auto_fusion/build/4e26b3ed4690a11c0fad61432f5a413b/external/llvm-raw/llvm/include/llvm/ADT/STLFunctionalExtras.h:45)
mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>, llvm::raw_ostream&, bool, bool) (Unknown Source:0)
mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::function_ref<mlir::LogicalResult (mlir::PassManager&)>, mlir::DialectRegistry&, bool, bool, bool, bool, bool, bool, bool)(llvm::raw_ostream & outputStream, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> > buffer, mlir::PassPipelineFn passManagerSetupFn, mlir::DialectRegistry & registry, bool splitInputFile, bool verifyDiagnostics, bool verifyPasses, bool allowUnregisteredDialects, bool preloadDialectsInContext, bool emitBytecode, bool implicitModule) (/data00/home/son.nguyen/workspace/auto_fusion_dev/erdos/operators/auto_fusion/build/4e26b3ed4690a11c0fad61432f5a413b/external/llvm-raw/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:187)
mlir::MlirOptMain(llvm::raw_ostream & outputStream, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> > buffer, const mlir::PassPipelineCLParser & passPipeline, mlir::DialectRegistry & registry, bool splitInputFile, bool verifyDiagnostics, bool verifyPasses, bool allowUnregisteredDialects, bool preloadDialectsInContext, bool emitBytecode, bool implicitModule, bool dumpPassPipeline) (/data00/home/son.nguyen/workspace/auto_fusion_dev/erdos/operators/auto_fusion/build/4e26b3ed4690a11c0fad61432f5a413b/external/llvm-raw/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:212)
mlir::MlirOptMain(int argc, char ** argv, llvm::StringRef toolName, mlir::DialectRegistry & registry, bool preloadDialectsInContext) (/data00/home/son.nguyen/workspace/auto_fusion_dev/erdos/operators/auto_fusion/build/4e26b3ed4690a11c0fad61432f5a413b/external/llvm-raw/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:306)
main(int argc, char ** argv) (/data00/home/son.nguyen/workspace/auto_fusion_dev/erdos/operators/auto_fusion/tools/kernel-gen-opt.cc:55)
```

# Operation Pass
https://mlir.llvm.org/docs/PassManagement/#operation-pass  
All passes in MLIR derive from `OperationPass`. 