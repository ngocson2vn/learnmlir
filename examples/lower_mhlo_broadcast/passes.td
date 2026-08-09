include "mlir/Pass/PassBase.td"

def CustomComputeOpAndFuncBufferizePass: Pass<"custom-bufferize", "mlir::ModuleOp"> {
    let summary = "Custom bufferize pass";
    let description = [{
      This pass converts tensors to memref buffers.
    }];
}
