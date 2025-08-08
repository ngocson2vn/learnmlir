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
MLIR also supports general OperationPasses that run on any isolated operation. All passes in MLIR derive from `OperationPass`.  
Implementing a pass is done by creating a class inheriting from `mlir::OperationPass<FuncOp>` and overriding the `runOnOperation()` method.  
For example, https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/  
```C++
  void runOnOperation() override {
    Operation *op = getOperation();
    resetIndent();
    printOperation(op);
  }
```

# Output file
```C++
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

std::string errorMessage;
auto output = mlir::openOutputFile("/data00/son.nguyen/log/ops.log", &errorMessage);
if (!output) {
  llvm::errs() << errorMessage << "\n";
  return std::terminate();
}

llvm::raw_ostream& os = output->os();
os << op->getName();
output->keep();
```

# Block
```C++
llvm-project/mlir/include/mlir/IR/Block.h
```

# Debug
-debug
--mlir-print-debuginfo
```C++
// main.cc
#include "llvm/Support/Debug.h"

LLVM_DEBUG(llvm::dbgs() << "Start initializing " << getDialectNamespace() << "\n");
```

build.sh:
```Bash
COPTS="-O0,-g,-fno-inline,-UNDEBUG"
SRC_FILES=+toyc.cc
SRC_FILES=${SRC_FILES},+llvm/lib/Support/Debug.cpp
SRC_FILES=${SRC_FILES},+llvm/lib/Support/CommandLine.cpp

CC=/usr/bin/gcc ./bazel --output_user_root=./build build -s //:toyc --config=cuda --per_file_copt=${SRC_FILES}@${COPTS} --strip=never $FLAGS -j 128
```

# Dump Clusters
```C++
// lib/Transforms/cluster_algo.h
const std::vector<Operation *>& getCluster() const;

// lib/Transforms/cluster_algo.cc
const std::vector<Operation *>& getCluster() const {
  return cluster;
}

// lib/Transforms/fuse_cwise_ops.cc
{
  std::error_code errCode;
  llvm::raw_fd_ostream outputFile("before_merge_clusters.mlir", errCode);
  if (errCode.value()) {
    std::terminate();
  }

  for (auto& c : clusters.getClusters()) {
    for (auto& op : c.getCluster()) {
      outputFile << *op << "\n";
    }
    outputFile << "\n\n";
    outputFile.flush();
  }
  outputFile.close();
  llvm::outs() << "Outputted before_merge_clusters.mlir\n\n";
}
```

# Dump Ops
```C++
  std::error_code errCode;
  llvm::raw_fd_ostream outputFile(output_fp, errCode);
  if (errCode.value()) {
    std::abort();
  }

  for (auto op : cluster) {
    outputFile << *op << "\n";
  }

  outputFile.close();
```

# Dump FuncOp V1
```C++

```

# Dump FuncOp V2
```C++
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

std::string errorMessage;
auto output = mlir::openOutputFile("layer_norm.mlir", &errorMessage);
if (!output) {
  llvm::errs() << errorMessage << "\n";
  std::terminate();
}
llvm::raw_ostream& os = output->os();

auto returnOp = cast<func::ReturnOp>(func.getBody().front().getTerminator());

if (returnOp.getOperands().size() != 1) {
  os << *func << "\n\n";
  os << *returnOp << "\n";
  output->keep();
}
```

# Dump FuncOp V3
```C++
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
struct LayerNormConvert : public OpRewritePattern<func::FuncOp>,
                          ConvertBase<LayerNormConvert> {
  using OpRewritePattern::OpRewritePattern;
 public:
  static std::unique_ptr<llvm::ToolOutputFile>& getOutputStream() {
    std::string errorMessage;
    std::string output_filename = "layer_norm.mlir";
    static std::unique_ptr<llvm::ToolOutputFile> output = mlir::openOutputFile(output_filename, &errorMessage);
    if (!output) {
      llvm::errs() << errorMessage << "\n";
      std::terminate();
    }
    return output;
  }

  // Client code
  auto& output = LayerNormConvert::getOutputStream();
  llvm::raw_ostream& os = output->os();
  if (cluster.size() > 0) {
    os << "# ========================================================================================\n";
    for (auto op : cluster) {
      os << *op << "\n";
    }
    os << "\n";
    output->keep();
  }

```


# Pass Manager
## Dump IRs
```C++
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"  // from @llvm-project

  std::string errorMessage;

  // predict_online_13
  auto predict_online_13 = mlir::openOutputFile("predict_online_13.mlir", &errorMessage);
  if (!predict_online_13) {
    llvm::errs() << errorMessage << "\n";
    std::terminate();
  }

  mlir::OpPrintingFlags flag{};
  pm.enableIRPrinting(
    /*shouldPrintBeforePass=*/[](mlir::Pass* p, mlir::Operation* op) {
      if (op->getName().getStringRef() == "func.func") {
        auto funcOp = llvm::dyn_cast<FuncOp>(op);
        if (funcOp.getSymName() == "predict_online_13") {
          return true;
        }
      }

      return false;
    },
    /*shouldPrintAfterPass=*/[](mlir::Pass* p, mlir::Operation * op) {
      if (op->getName().getStringRef() == "func.func") {
        auto funcOp = llvm::dyn_cast<FuncOp>(op);
        if (funcOp.getSymName() == "predict_online_13") {
          return true;
        }
      }

      return false;
    },
    /*printModuleScope=*/false, 
    /*printAfterOnlyOnChange=*/false,
    /*printAfterOnlyOnFailure=*/false, 
    predict_online_13->os(), flag
  );

  // predict_online_14
  auto predict_online_14 = mlir::openOutputFile("predict_online_14.mlir", &errorMessage);
  if (!predict_online_14) {
    llvm::errs() << errorMessage << "\n";
    std::terminate();
  }

  pm.enableIRPrinting(
    /*shouldPrintBeforePass=*/[](mlir::Pass* p, mlir::Operation* op) {
      if (op->getName().getStringRef() == "func.func") {
        auto funcOp = llvm::dyn_cast<FuncOp>(op);
        if (funcOp.getSymName() == "predict_online_14") {
          return true;
        }
      }

      return false;
    },
    /*shouldPrintAfterPass=*/[](mlir::Pass* p, mlir::Operation * op) {
      if (op->getName().getStringRef() == "func.func") {
        auto funcOp = llvm::dyn_cast<FuncOp>(op);
        if (funcOp.getSymName() == "predict_online_14") {
          return true;
        }
      }

      return false;
    },
    /*printModuleScope=*/false, 
    /*printAfterOnlyOnChange=*/false,
    /*printAfterOnlyOnFailure=*/false, 
    predict_online_14->os(), flag
  );

  ...

  predict_online_13->keep();
  predict_online_14->keep();
```

# Get pass name
```C++
// llvm-project/mlir/lib/Pass/Pass.cpp
LogicalResult OpToOpPassAdaptor::run(Pass *pass, Operation *op,
                                     AnalysisManager am, bool verifyPasses,
                                     unsigned parentInitGeneration) {
  // Omit
}

-exec p pass->getName()
```

## Disable multithreading
```C++
  auto ctx = module.getContext();
  ctx->enableMultithreading(false);
  mlir::PassManager pm(ctx);
```

# Common Errors
## error: operand #0 does not dominate this use
Solution:  
Modify llvm-project/mlir/lib/IR/Verifier.cpp to print out the faulty op:
```C++
/// Emit an error when the specified operand of the specified operation is an
/// invalid use because of dominance properties.
static void diagnoseInvalidOperandDominance(Operation &op, unsigned operandNo) {
  InFlightDiagnostic diag = op.emitError("operand #")
                            << operandNo << " does not dominate this use: "
                            << op;
```

## error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: builtin.module



# Location to string
```C++
static std::string locationToString(mlir::Location loc) {
  std::string storage;
  llvm::raw_string_ostream oss(storage);
  loc.print(oss);
  return storage;
}
```

# rewriter.create
```C+++
    auto subViewOp = rewriter.create<memref::SubViewOp>(
        op.getLoc(), adaptor.input(), subviewOffsets, subviewSizes,
        subviewStrides);
```
<br/>

-> llvm-project/mlir/include/mlir/IR/Builders.h
```C++
  /// Create an operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    OperationState state(location,
                         getCheckRegisteredInfo<OpTy>(location.getContext()));
    OpTy::build(*this, state, std::forward<Args>(args)...);
    auto *op = create(state);
    auto result = dyn_cast<OpTy>(op);
    assert(result && "builder didn't return the right type");
    return result;
  }
```
-> llvm-project/mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp

# PassRegistry
llvm-project/mlir/lib/Pass/PassRegistry.cpp
```C++
/// Static mapping of all of the registered passes.
static llvm::ManagedStatic<llvm::StringMap<PassInfo>> passRegistry;

void mlir::registerPass(const PassAllocatorFunction &function);
```

# OpInterface
Details: [interface.cc](./interface.cc)
>The `OpInterface` template allows operations to share common functionality (e.g., methods like `getResultBuffer`) while remaining decoupled from a single inheritance hierarchy. 
Because an operation inherits a specialized version of the `OpInterface` template which is always an unique one. For example,
```C++
// examples/bazel-out/k8-dbg/bin/external/org_tensorflow/tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_structured_interface.h.inc
class LmhloOp : public ::mlir::OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits> {
  // Omit for brevity
}

struct LmhloOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    Value (*getResultBuffer)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::lmhlo::LmhloOp;
    Model() : Concept{getResultBuffer} {}

    static inline Value getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::lmhlo::LmhloOp;
    FallbackModel() : Concept{getResultBuffer} {}

    static inline Value getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    Value getResultBuffer(::mlir::Operation *tablegen_opaque_val) const;
  };
};

template<typename ConcreteOp>
Value detail::LmhloOpInterfaceTraits::Model<ConcreteOp>::getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResultBuffer();
}
template<typename ConcreteOp>
Value detail::LmhloOpInterfaceTraits::FallbackModel<ConcreteOp>::getResultBuffer(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getResultBuffer(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::LmhloOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getResultBuffer(::mlir::Operation *tablegen_opaque_val) const {
/// By default, the result buffer is the last operand 
        unsigned num_operands = this->getOperation()->getNumOperands();
        if (num_operands > 1) {
          return this->getOperation()->getOperand(num_operands - 1);
        }
        return nullptr;
}


// OpInterface
class OpInterface<LmhloOp, detail::LmhloOpInterfaceTraits> : public detail::Interface<LmhloOp, Operation *, detail::LmhloOpInterfaceTraits, Op<LmhloOp>, OpTrait::TraitBase>


// Interface
class Interface<LmhloOp, Operation*, detail::LmhloOpInterfaceTraits, Op<LmhloOp>, OpTrait::TraitBase> : public Op<LmhloOp> {
public:
  template <typename T>
  using Model = typename detail::LmhloOpInterfaceTraits::template Model<T>;

  /// This is a special trait that registers a given interface with an object.
  template <typename ConcreteT>
  struct Trait : public OpTrait::TraitBase<ConcreteT, Trait> {
    using ModelT = Model<ConcreteT>;

    /// Define an accessor for the ID of this interface.
    static TypeID getInterfaceID() { return TypeID::get<LmhloOp>(); }
  };
}


// OpTrait::TraitBase
template <typename ConcreteType, template <typename> class TraitType>
class TraitBase {
protected:
  /// Return the ultimate Operation being worked on.
  Operation *getOperation() {
    auto *concrete = static_cast<ConcreteType *>(this);
    return concrete->getOperation();
  }
};
```
