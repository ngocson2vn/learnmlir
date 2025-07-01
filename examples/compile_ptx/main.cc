#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/asm_compiler.h"

#include <iostream>
#include <fstream>

using namespace mlir;

int main(int argc, char** argv) {
  // Initialize MLIR context and register dialects
  MLIRContext context;

  static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input ptx file>"),
    llvm::cl::init("-"),
    llvm::cl::value_desc("filename")
  );

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "compile_ptx\n");


  std::ifstream is(inputFilename.getValue().c_str(), std::ios::in);
  if (!is.is_open() || !is.good()) {
    llvm::errs() << "Failed to open input file " << inputFilename << "\n";
    return 1;
  }

  // get length of file:
  is.seekg (0, is.end);
  int length = is.tellg();
  is.seekg (0, is.beg);

  char* ptx = new char[length];
  is.read(ptx, length);

  xla::HloModuleConfig config;
  xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
  options.set_xla_gpu_ftz(false);
  options.set_xla_gpu_dump_llvmir(false);

  // Make sure we use full precision division operations.
  (*options.mutable_xla_backend_extra_options())["-nvptx-prec-divf32"] = "2";
  (*options.mutable_xla_backend_extra_options())["-simplifycfg-sink-common"] = "false";
  config.set_debug_options(options);
  int cc_major = 8;
  int cc_minor = 0;
  auto gpu_asm_opts = xla::gpu::PtxOptsFromDebugOptions(config.debug_options());
  auto gpu_asm = tensorflow::se::CompileGpuAsm(cc_major, cc_minor, ptx, gpu_asm_opts);
  if (!gpu_asm.ok()) {
    llvm::errs() << gpu_asm.status().error_message().c_str() << "\n";
    return 1;
  } else {
    llvm::outs() << "Sucess" << "\n";
  }

  return 0;
}
