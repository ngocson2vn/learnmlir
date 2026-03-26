# FuncOpConverter
Output:
```MLIR
Begin FuncOpConverter:
module {
  func.func @main(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>) -> tensor<?xf64> {
    %0 = arith.addf %arg0, %arg1 : tensor<?xf64>
    return %0 : tensor<?xf64>
  }
}

After FuncOpConverter:
type of return operand 0 ('tensor<?xf64>') doesn't match function result type ('memref<?xf64>') in function @main
'func.func' op symbol declaration cannot have public visibility
[mlir-asm-printer AsmPrinter.cpp:2073 1] builtin.module' failed to verify and will be printed in generic form
"builtin.module"() ({
  "func.func"() <{function_type = (memref<?xf64>, memref<?xf64>) -> memref<?xf64>, sym_name = "main"}> ({
  ^bb0(%arg0: memref<?xf64>, %arg1: memref<?xf64>):
    %0 = "arith.addf"(<<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>) <{fastmath = #arith.fastmath<none>}> : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    "func.return"(%0) : (tensor<?xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>, sym_name = "main"}> ({
  }) : () -> ()
}) : () -> ()
```

## Why `<<UNKNOWN SSA VALUE>>` happens
When you call `rewriter.applySignatureConversion(&entryBlock, ...)`, the rewriter does the following:
* It creates the **new** block arguments of type `memref<?xf64>`.
* It removes the **old** block arguments of type `tensor<?xf64>` from the block.
* It logs internally that any uses of the old arguments should be replaced by the new arguments (often via the source materializations you defined).

However, at the moment you call `LLVM_DEBUG(llvm::dbgs() << ... << *newFunc->getParentOp())`, the `arith.addf` operation has not been updated yet. Its operands are still pointing to the **old** `tensor` block arguments. 

Because those old block arguments have been detached from the block by the signature conversion, they are floating in memory without a parent. When the MLIR ASM Printer tries to print the `arith.addf` op, it looks at the operands, tries to trace them back to their parent block to print their names (like `%arg0`), fails to find a parent block, and defaults to `<<UNKNOWN SSA VALUE>>`.

