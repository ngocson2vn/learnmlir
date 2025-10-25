class HostModuleToObject {
 public:
  HostModuleToObject(ModuleOp& moduleOp,
                     StringRef triple,
                     StringRef chip,
                     StringRef features = {}, 
                     int optLevel = 3) 
    : moduleOp(moduleOp),
      triple(triple),
      chip(chip),
      features(features),
      optLevel(optLevel) {}

  LogicalResult emitObjectFile(const std::string& objectFilePath) {
    // Translate the module to LLVM IR.
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(moduleOp, llvmContext);
    if (!llvmModule) {
      moduleOp.emitError() << "Failed creating the llvm::Module.";
      return failure();
    }

    setDataLayoutAndTriple(*llvmModule);

    // Optimize the module.
    if (failed(optimizeModule(*llvmModule, optLevel))) {
      return failure();
    }

    std::string objectStr;
    llvm::raw_string_ostream stream(objectStr);
    auto& targetMachine = *getOrCreateTargetMachine().value();

    { // Drop pstream after this to prevent the ISA from being stuck buffering
      llvm::buffer_ostream pstream(stream);
      llvm::legacy::PassManager codegenPasses;

      if (targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                            llvm::CodeGenFileType::ObjectFile))
        return failure();

      if (!codegenPasses.run(*llvmModule)) {
        return failure();
      }
    }

    std::ofstream ofs(objectFilePath, std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
      llvm::errs() << "Failed to open objectFilePath " << objectFilePath << "\n";
      return failure();
    }

    ofs.write(objectStr.c_str(), objectStr.size());
    ofs.flush();
    ofs.close();

    return success();
  }

 private:
  std::optional<llvm::TargetMachine *>
  getOrCreateTargetMachine() {
    if (targetMachine)
      return targetMachine.get();
    // Load the target.
    std::string error;
    llvm::Triple parsedTriple(triple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(parsedTriple, error);
    if (!target) {
      moduleOp.emitError()
          << "Failed to lookup target for triple '" << triple << "' " << error;
      return std::nullopt;
    }

    // Create the target machine using the target.
    llvm::TargetOptions targetOptions =
        llvm::codegen::InitTargetOptionsFromCodeGenFlags(parsedTriple);
    targetMachine.reset(
        target->createTargetMachine(parsedTriple,
                                    chip,
                                    features,
                                    targetOptions,
                                    llvm::Reloc::Model::PIC_));
    if (!targetMachine)
      return std::nullopt;
    return targetMachine.get();
  }

  void setDataLayoutAndTriple(llvm::Module &module) {
    // Create the target machine.
    std::optional<llvm::TargetMachine *> targetMachine =
        getOrCreateTargetMachine();
    if (targetMachine) {
      // Set the data layout and target triple of the module.
      module.setDataLayout((*targetMachine)->createDataLayout());
      module.setTargetTriple((*targetMachine)->getTargetTriple());
    }
  }

  LogicalResult optimizeModule(llvm::Module &module, int optLevel) {
    if (optLevel < 0 || optLevel > 3)
      return moduleOp.emitError()
            << "Invalid optimization level: " << optLevel << ".";

    std::optional<llvm::TargetMachine *> targetMachine =
        getOrCreateTargetMachine();
    if (!targetMachine)
      return moduleOp.emitError()
            << "Target Machine unavailable for triple " << triple
            << ", can't optimize with LLVM\n";
    (*targetMachine)->setOptLevel(static_cast<llvm::CodeGenOptLevel>(optLevel));

    auto transformer =
        makeOptimizingTransformer(optLevel, /*sizeLevel=*/0, *targetMachine);
    auto error = transformer(&module);
    if (error) {
      InFlightDiagnostic mlirError = moduleOp.emitError();
      llvm::handleAllErrors(
          std::move(error), [&mlirError](const llvm::ErrorInfoBase &ei) {
            mlirError << "Could not optimize LLVM IR: " << ei.message() << "\n";
          });
      return mlirError;
    }
    return success();
  }

 private:
  ModuleOp& moduleOp;
  StringRef triple;
  StringRef chip;
  StringRef features;
  int optLevel;
  std::unique_ptr<llvm::TargetMachine> targetMachine;
};
