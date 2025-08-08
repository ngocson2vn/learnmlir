include "mlir/Pass/PassBase.td"

def ConvertToyToArith: Pass<"convert-toy-to-arith", "mlir::ModuleOp"> {
    let summary = "Convert Toy to Arith";
    let description = [{
      This pass converts the Toy Dialect into the Arith Dialect.
    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::toy::ToyDialect"];
}

def ConvertTensorToMemRef: Pass<"convert-tensor-to-memref", "mlir::ModuleOp"> {
    let summary = "Convert Tensor to MemRef";
    let description = [{
      This pass converts the Tensor type into the MemRef type.
    }];

    let dependentDialects = ["mlir::arith::ArithDialect"];
}
