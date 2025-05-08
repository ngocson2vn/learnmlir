include "mlir/IR/OpBase.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "mlir::toy";
}

def Toy_AddOp : Op<Toy_Dialect, "add"> {
  let summary = "Addition operation for the toy dialect";
  let description = [{
    This operation computes the sum of two tensors element-wise.
  }];
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict `->` type($result)";
}
