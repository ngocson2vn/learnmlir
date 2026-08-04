#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"

#include <unordered_map>

llvm::raw_fd_ostream& getDebugOs(const char* output_file_name) {
  std::error_code errCode;
  static std::unordered_map<std::string, llvm::raw_fd_ostream> os_map;
  auto it = os_map.find(output_file_name);
  if (it != os_map.end()) {
    return it->second;
  }
  auto res = os_map.try_emplace(output_file_name, output_file_name, errCode);
  if (!res.second) {
    std::terminate();
  }

  auto& node = *res.first;
  return node.second;
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
