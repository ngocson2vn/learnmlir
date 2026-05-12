# MLIR print
```C++
// The OpPrintingFlags().useGenericFormat() flag forces the generic syntax over the shorthand (Format 2).

// Format 1: %0 = "arith.constant"() {value = 0 : index} : () -> index  
// Uses the generic form with an operation name in quotes, attributes in {} (e.g., value), and an explicit function-like type signature () -> index.

// Format 2: %c0 = arith.constant 0 : index  
// Uses the custom, human-readable shorthand form, omitting the attribute name (value) and type signature, making it more concise.


// Print ops with locs
// mlir::OpPrintingFlags flag = mlir::OpPrintingFlags().enableDebugInfo(true);

ModuleOp mod = getOperation();
mod.print(llvm::outs(), OpPrintingFlags().printGenericOpForm(false));
```