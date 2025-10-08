#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"

static llvm::raw_fd_ostream& getDebugOs(const char* output_file_name) {
  std::error_code errCode;
  static llvm::raw_fd_ostream debugOs(output_file_name, errCode);
  if (errCode.value()) {
    std::terminate();
  }

  return debugOs;
}

int main(int argc, char** argv) {
  std::string errorMessage;

  // lowering
  auto output = mlir::openOutputFile("lowering.mlir", &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    std::terminate();
  }
  output->keep();

  mlir::OpPrintingFlags flag{};
  pm.enableIRPrinting(
    /*shouldPrintBeforePass=*/[](mlir::Pass* p, mlir::Operation* op) {
      return false;
    },
    /*shouldPrintAfterPass=*/[](mlir::Pass* p, mlir::Operation * op) {
      return true;
    },
    /*printModuleScope=*/false, 
    /*printAfterOnlyOnChange=*/true,
    /*printAfterOnlyOnFailure=*/false, 
    output->os(), flag
  );
}
