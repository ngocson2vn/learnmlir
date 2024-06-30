
# PrintNestingPass
https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/

## Call Stack
```C++
(anonymous namespace)::PrintNestingPass::runOnOperation((anonymous namespace)::PrintNestingPass * this) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/pass1/mlir/traverse.cc:24)

mlir::detail::OpToOpPassAdaptor::run(mlir::Pass * pass, mlir::Operation * op, mlir::AnalysisManager am, bool verifyPasses, unsigned int parentInitGeneration) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/mlir/lib/Pass/Pass.cpp:461)

mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager & pm, mlir::Operation * op, mlir::AnalysisManager am, bool verifyPasses, unsigned int parentInitGeneration, mlir::PassInstrumentor * instrumentor, const mlir::PassInstrumentation::PipelineParentInfo * parentInfo) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/mlir/lib/Pass/Pass.cpp:525)

mlir::PassManager::runPasses(mlir::PassManager * this, mlir::Operation * op, mlir::AnalysisManager am) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/mlir/lib/Pass/Pass.cpp:828)

mlir::PassManager::run(mlir::PassManager * this, mlir::Operation * op) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/mlir/lib/Pass/Pass.cpp:808)

performActions(llvm::raw_ostream&, bool, bool, llvm::SourceMgr&, mlir::MLIRContext*, llvm::function_ref<mlir::LogicalResult (mlir::PassManager&)>, bool, bool)(llvm::raw_ostream & os, bool verifyDiagnostics, bool verifyPasses, llvm::SourceMgr & sourceMgr, mlir::MLIRContext * context, mlir::PassPipelineFn passManagerSetupFn, bool emitBytecode, bool implicitModule) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:91)

processBuffer(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, bool, bool, bool, bool, bool, bool, llvm::function_ref<mlir::LogicalResult (mlir::PassManager&)>, mlir::DialectRegistry&, llvm::ThreadPool*)(llvm::raw_ostream & os, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> > ownedBuffer, bool verifyDiagnostics, bool verifyPasses, bool allowUnregisteredDialects, bool preloadDialectsInContext, bool emitBytecode, bool implicitModule, mlir::PassPipelineFn passManagerSetupFn, mlir::DialectRegistry & registry, llvm::ThreadPool * threadPool) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:139)

mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::function_ref<mlir::LogicalResult (mlir::PassManager&)>, mlir::DialectRegistry&, bool, bool, bool, bool, bool, bool, bool)::$_0::operator()(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&) const(const class {...} * this, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> > chunkBuffer, llvm::raw_ostream & os) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:181)

llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::function_ref<mlir::LogicalResult (mlir::PassManager&)>, mlir::DialectRegistry&, bool, bool, bool, bool, bool, bool, bool)::$_0>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)(intptr_t callable, llvm::raw_ostream & params, llvm::raw_ostream & params) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:45)

llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>::operator()(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&) const(const llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream &)> * this, llvm::raw_ostream & params, llvm::raw_ostream & params) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/llvm/include/llvm/ADT/STLFunctionalExtras.h:68)

mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::raw_ostream&)>, llvm::raw_ostream&, bool, bool)(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> > originalBuffer, mlir::ChunkBufferHandler processChunkBuffer, llvm::raw_ostream & os, bool enableSplitting, bool insertMarkerInOutput) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/mlir/lib/Support/ToolUtilities.cpp:28)

mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> >, llvm::function_ref<mlir::LogicalResult (mlir::PassManager&)>, mlir::DialectRegistry&, bool, bool, bool, bool, bool, bool, bool)(llvm::raw_ostream & outputStream, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> > buffer, mlir::PassPipelineFn passManagerSetupFn, mlir::DialectRegistry & registry, bool splitInputFile, bool verifyDiagnostics, bool verifyPasses, bool allowUnregisteredDialects, bool preloadDialectsInContext, bool emitBytecode, bool implicitModule) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:186)

mlir::MlirOptMain(llvm::raw_ostream & outputStream, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer> > buffer, const mlir::PassPipelineCLParser & passPipeline, mlir::DialectRegistry & registry, bool splitInputFile, bool verifyDiagnostics, bool verifyPasses, bool allowUnregisteredDialects, bool preloadDialectsInContext, bool emitBytecode, bool implicitModule, bool dumpPassPipeline) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:209)

mlir::MlirOptMain(int argc, char ** argv, llvm::StringRef toolName, mlir::DialectRegistry & registry, bool preloadDialectsInContext) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/external/llvm-project/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp:306)

main(int argc, char ** argv) (/data00/home/son.nguyen/workspace/learnmlir/toy/bazel-toy/pass1/main.cc:17)
```