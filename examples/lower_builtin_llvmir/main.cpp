#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"


using namespace mlir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // This registration is required for translating builtin.module op
  // Otherwise, MLIR will report "error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: builtin.module"
  mlir::registerBuiltinDialectTranslation(registry);

  // This registration is required for translating LLVM dialect ops
  mlir::registerLLVMDialectTranslation(registry);

  // Set up the MLIR context
  MLIRContext context(registry);
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<LLVM::LLVMDialect>();

  // Build a sample ModuleOp
  OpBuilder builder(&context);
  auto loc = UnknownLoc::get(&context);
  ModuleOp moduleOp = builder.create<ModuleOp>(loc);
  builder.setInsertionPointToStart(moduleOp.getBody());

  auto i32Type = IntegerType::get(&context, 32);
  auto funcType = LLVM::LLVMFunctionType::get(&context, i32Type, {i32Type}, false);
  auto mainFunc = builder.create<LLVM::LLVMFuncOp>(loc, "main", funcType);
  Block* mainBody = mainFunc.addEntryBlock(builder);
  builder.setInsertionPointToStart(mainBody);
  Value arg0 = mainFunc.getArgument(0);
  Value res = builder.create<LLVM::AddOp>(loc, builder.getI32Type(), arg0, arg0);
  builder.create<LLVM::ReturnOp>(loc, ValueRange{res});

  llvm::outs() << "=================================================\n";
  llvm::outs() << "Original moduleOp\n";
  llvm::outs() << "=================================================\n";
  moduleOp->dump();
  llvm::outs() << "=================================================\n";
  llvm::outs() << "\n";

  // Lower to LLVM module.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(moduleOp, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Could not translate MLIR module to LLVM IR" << "\n";
    return 1;
  }
  llvmModule->setModuleIdentifier("sony");

  // Dump LLVM IR
  llvm::outs() << "=================================================\n";
  llvm::outs() << "LLVM IR\n";
  llvm::outs() << "=================================================\n";
  llvmModule->dump();
  llvm::outs() << "=================================================\n";

  // Optionally save to file
  auto output = openOutputFile("./output.mlir");
  if (!output) {
    llvm::errs() << "Failed to open output file\n";
    return 1;
  }
  output->keep();
  llvmModule->print(output->os(), nullptr);

  return 0;
}
