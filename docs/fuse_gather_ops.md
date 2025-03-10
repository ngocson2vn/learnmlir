Please generalize for the following pattern:
- Multiple params with the same shape
- Multiple indices with the same shape
- The same axis value.


To generalize the implementation for the pattern you’ve specified—multiple params with the same shape, multiple indices with the same shape, and the same axis value—we’ll adapt the GPU-accelerated func::FuncOp to handle this scenario efficiently. This pattern might arise when you’re performing gathers on multiple input tensors (e.g., batches or channels) with corresponding index tensors, all along a consistent axis. The solution will use MLIR’s GPU dialect to parallelize the computation across these multiple inputs, ensuring scalability and performance on GPU devices.

## Overview
- **Pattern:**
  - params: Multiple tensors with the same shape (e.g., tensor<?x?xf32>).
  - indices: Multiple tensors with the same shape (e.g., tensor<?xi32>), one per params.
  - axis: A single scalar tensor (e.g., tensor<i32>), shared across all gathers.
- **Goal:** Fuse these into a single func::FuncOp that performs all gathers in parallel on the GPU, producing one output tensor per params/indices pair.
- **Approach:**
  - Use a single gpu.launch kernel to process all params and indices pairs.
  - Map threads to output elements and use an additional dimension (e.g., block index) to iterate over the multiple inputs.
  - Handle dynamic shapes and arbitrary axis values.

## Steps to Generalize for Multiple params and indices
### 1. Define the Generalized GPU Fused Function
The function will:
- Take a list of params tensors, a list of indices tensors, and a single axis tensor as inputs.
- Allocate output buffers for each params/indices pair.
- Use a single GPU kernel to parallelize across all pairs, leveraging block and thread indices.

Here’s the generalized MLIR output:
```mlir
func.func @fused_gather(%params0: tensor<?x?xf32>, %params1: tensor<?x?xf32>,
                        %indices0: tensor<?xi32>, %indices1: tensor<?xi32>,
                        %axis: tensor<i32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  // Extract dynamic dimensions (assume all params and indices have the same shape)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %params_dim0 = tensor.dim %params0, %c0 : tensor<?x?xf32>
  %params_dim1 = tensor.dim %params0, %c1 : tensor<?x?xf32>
  %indices_dim0 = tensor.dim %indices0, %c0 : tensor<?xi32>
  %num_pairs = arith.constant 2 : index  // Number of params/indices pairs
  %axis_val = tensor.extract %axis[] : tensor<i32>

  // Convert tensors to GPU memrefs
  %params0_mem = memref.buffer_cast %params0 : tensor<?x?xf32> to memref<?x?xf32, #gpu.address_space<global>>
  %params1_mem = memref.buffer_cast %params1 : tensor<?x?xf32> to memref<?x?xf32, #gpu.address_space<global>>
  %indices0_mem = memref.buffer_cast %indices0 : tensor<?xi32> to memref<?xi32, #gpu.address_space<global>>
  %indices1_mem = memref.buffer_cast %indices1 : tensor<?xi32> to memref<?xi32, #gpu.address_space<global>>

  // Allocate output buffers
  %out0_mem = memref.alloc(%indices_dim0, %params_dim1) : memref<?x?xf32, #gpu.address_space<global>>
  %out1_mem = memref.alloc(%indices_dim0, %params_dim1) : memref<?x?xf32, #gpu.address_space<global>>

  // Single GPU kernel for all pairs
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %num_pairs, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %indices_dim0, %block_y = %params_dim1, %block_z = %c1) {
    // Indices for pair, row, and column
    %pair_idx = gpu.block_id x
    %i = gpu.thread_id x
    %j = gpu.thread_id y

    // Select params and indices based on pair_idx
    %params_mem = arith.select %pair_idx, %params0_mem, %params1_mem : index, memref<?x?xf32, #gpu.address_space<global>>
    %indices_mem = arith.select %pair_idx, %indices0_mem, %indices1_mem : index, memref<?xi32, #gpu.address_space<global>>
    %out_mem = arith.select %pair_idx, %out0_mem, %out1_mem : index, memref<?x?xf32, #gpu.address_space<global>>

    // Load index and gather value
    %idx = memref.load %indices_mem[%i] : memref<?xi32, #gpu.address_space<global>>
    %val = scf.if %axis_val -> (f32) {
      %v = memref.load %params_mem[%idx, %j] : memref<?x?xf32, #gpu.address_space<global>>
      scf.yield %v : f32
    } else {
      %v = memref.load %params_mem[%j, %idx] : memref<?x?xf32, #gpu.address_space<global>>
      scf.yield %v : f32
    }
    memref.store %val, %out_mem[%i, %j] : memref<?x?xf32, #gpu.address_space<global>>

    gpu.terminator
  }

  // Convert back to tensors
  %out0 = memref.buffer_cast %out0_mem : memref<?x?xf32, #gpu.address_space<global>> to tensor<?x?xf32>
  %out1 = memref.buffer_cast %out1_mem : memref<?x?xf32, #gpu.address_space<global>> to tensor<?x?xf32>
  return %out0, %out1 : tensor<?x?xf32>, tensor<?x?xf32>
}
```

### 2. Update the Rewrite Pattern
Here’s the updated C++ code to handle multiple params and indices with the same shape and a shared axis:
```cpp
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace {

struct FuseGatherV2ToFuncPattern : public OpRewritePattern<TF::GatherV2Op> {
  using OpRewritePattern<TF::GatherV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::GatherV2Op op,
                                PatternRewriter &rewriter) const override {
    // Collect fusable GatherV2 ops with same axis and matching shapes.
    Value axis = op.getAxis();
    SmallVector<TF::GatherV2Op, 4> fusableOps = {op};
    SmallVector<Value, 4> paramsList = {op.getParams()};
    SmallVector<Value, 4> indicesList = {op.getIndices()};
    SmallVector<Type, 4> outputTypes = {op.getResult().getType()};

    auto paramsShape = op.getParams().getType().cast<TensorType>().getShape();
    auto indicesShape = op.getIndices().getType().cast<TensorType>().getShape();

    for (auto otherOp : op.getParentBlock()->getOps<TF::GatherV2Op>()) {
      if (otherOp == op) continue;
      auto otherParamsShape = otherOp.getParams().getType().cast<TensorType>().getShape();
      auto otherIndicesShape = otherOp.getIndices().getType().cast<TensorType>().getShape();
      if (otherOp.getAxis() == axis && otherOp.getBatchDims() == op.getBatchDims() &&
          otherParamsShape == paramsShape && otherIndicesShape == indicesShape) {
        fusableOps.push_back(otherOp);
        paramsList.push_back(otherOp.getParams());
        indicesList.push_back(otherOp.getIndices());
        outputTypes.push_back(otherOp.getResult().getType());
      }
    }

    if (fusableOps.size() < 2) return failure();

    // Create the fused function.
    auto module = op->getParentOfType<ModuleOp>();
    SymbolTable symbolTable(module);
    std::string funcName = "fused_gather_" + std::to_string(module.getOps<func::FuncOp>().size());
    OpBuilder funcBuilder(module.getBodyRegion());

    // Define function signature.
    SmallVector<Type, 4> inputTypes;
    inputTypes.append(paramsList.begin(), paramsList.end());
    inputTypes.append(indicesList.begin(), indicesList.end());
    inputTypes.push_back(axis.getType());
    FunctionType funcType = FunctionType::get(getContext(), inputTypes, outputTypes);
    auto funcOp = funcBuilder.create<func::FuncOp>(op.getLoc(), funcName, funcType);
    funcOp.setPrivate();
    Block* block = funcOp.addEntryBlock();
    rewriter.setInsertionPointToStart(block);

    // Constants.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    Value numPairs = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), paramsList.size());

    // Extract dynamic dimensions (all params and indices have the same shape).
    Value paramsDim0 = rewriter.create<tensor::DimOp>(op.getLoc(), paramsList[0], c0);
    Value paramsDim1 = rewriter.create<tensor::DimOp>(op.getLoc(), paramsList[0], c1);
    Value indicesDim0 = rewriter.create<tensor::DimOp>(op.getLoc(), indicesList[0], c0);
    Value axisVal = rewriter.create<tensor::ExtractOp>(op.getLoc(), rewriter.getI32Type(),
                                                       block->getArgument(inputTypes.size() - 1), ValueRange{});

    // Convert to GPU memrefs.
    auto gpuGlobal = gpu::AddressSpaceAttr::get(rewriter.getContext(), gpu::AddressSpace::Global);
    SmallVector<Value, 4> paramsMems, indicesMems;
    auto paramsMemType = MemRefType::get({-1, -1}, rewriter.getF32Type(), {}, gpuGlobal);
    auto indicesMemType = MemRefType::get({-1}, rewriter.getI32Type(), {}, gpuGlobal);
    for (int i = 0; i < paramsList.size(); ++i) {
      paramsMems.push_back(rewriter.create<memref::BufferCastOp>(op.getLoc(), paramsMemType, block->getArgument(i)));
      indicesMems.push_back(rewriter.create<memref::BufferCastOp>(op.getLoc(), indicesMemType,
                                                                 block->getArgument(i + paramsList.size())));
    }

    // Allocate GPU output buffers.
    SmallVector<Value, 4> outputMems;
    auto outMemType = MemRefType::get({-1, -1}, rewriter.getF32Type(), {}, gpuGlobal);
    for (int i = 0; i < paramsList.size(); ++i) {
      outputMems.push_back(rewriter.create<memref::AllocOp>(op.getLoc(), outMemType,
                                                            ValueRange{indicesDim0, paramsDim1}));
    }

    // Single GPU kernel for all pairs.
    auto launchOp = rewriter.create<gpu::LaunchOp>(
        op.getLoc(), numPairs, c1, c1, // Grid size: num_pairs x 1 x 1
        indicesDim0, paramsDim1, c1);  // Block size: indices_size x params_dim1 x 1
    rewriter.setInsertionPointToStart(launchOp.getBody());

    Value pairIdx = rewriter.create<gpu::BlockIdOp>(op.getLoc(), rewriter.getIndexType(), gpu::Dimension::x);
    Value tx = rewriter.create<gpu::ThreadIdOp>(op.getLoc(), rewriter.getIndexType(), gpu::Dimension::x);
    Value ty = rewriter.create<gpu::ThreadIdOp>(op.getLoc(), rewriter.getIndexType(), gpu::Dimension::y);

    // Select params, indices, and output based on pair_idx.
    Value paramsMem, indicesMem, outMem;
    for (int i = 0; i < paramsList.size(); ++i) {
      Value pairConst = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), i);
      Value cond = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::eq, pairIdx, pairConst);
      paramsMem = i == 0 ? paramsMems[i] : rewriter.create<arith::SelectOp>(op.getLoc(), cond, paramsMems[i], paramsMem);
      indicesMem = i == 0 ? indicesMems[i] : rewriter.create<arith::SelectOp>(op.getLoc(), cond, indicesMems[i], indicesMem);
      outMem = i == 0 ? outputMems[i] : rewriter.create<arith::SelectOp>(op.getLoc(), cond, outputMems[i], outMem);
    }

    // Gather logic.
    Value idx = rewriter.create<memref::LoadOp>(op.getLoc(), indicesMem, ValueRange{tx});
    auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), rewriter.getF32Type(), axisVal);
    {
      OpBuilder thenBuilder = ifOp.getThenBuilder();
      Value val = thenBuilder.create<memref::LoadOp>(op.getLoc(), paramsMem, ValueRange{idx, ty});
      thenBuilder.create<scf::YieldOp>(op.getLoc(), val);
    }
    {
      OpBuilder elseBuilder = ifOp.getElseBuilder();
      Value val = elseBuilder.create<memref::LoadOp>(op.getLoc(), paramsMem, ValueRange{ty, idx});
      elseBuilder.create<scf::YieldOp>(op.getLoc(), val);
    }
    rewriter.create<memref::StoreOp>(op.getLoc(), ifOp.getResult(0), outMem, ValueRange{tx, ty});
    rewriter.create<gpu::TerminatorOp>(op.getLoc());

    // Convert back to tensors.
    SmallVector<Value, 4> results;
    for (int i = 0; i < outputMems.size(); ++i) {
      auto tensor = rewriter.create<memref::BufferCastOp>(op.getLoc(), outputTypes[i], outputMems[i]);
      results.push_back(tensor);
    }
    rewriter.create<func::ReturnOp>(op.getLoc(), results);

    // Insert function and replace with call.
    symbolTable.insert(funcOp);
    SmallVector<Value, 4> callArgs;
    callArgs.append(paramsList.begin(), paramsList.end());
    callArgs.append(indicesList.begin(), indicesList.end());
    callArgs.push_back(axis);
    auto callOp = rewriter.create<func::CallOp>(op.getLoc(), funcOp, callArgs);
    for (size_t i = 0; i < fusableOps.size(); ++i) {
      rewriter.replaceOp(fusableOps[i], callOp.getResult(i));
    }

    return success();
  }
};

void populateGatherV2ToFuncPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FuseGatherV2ToFuncPattern>(context);
}

} // namespace
} // namespace mlir
```

### 3. Key Generalizations
- Multiple params and indices: The pattern now collects pairs of params and indices, ensuring they have matching shapes.
- Single Kernel: A single gpu.launch processes all pairs, using gpu.block_id x to select the pair and gpu.thread_id x/y for row/column indices.
- Dynamic Shapes: Dimensions are extracted at runtime, assuming all params share the same shape and all indices share the same shape.
- Axis Consistency: The shared axis is applied uniformly across all gathers.
- Selection Logic: arith.select dynamically chooses the correct params, indices, and output buffers based on the block index.

### 4. Build and Lower
Build TensorFlow with GPU support and run the pass:
```bash
tensorflow-opt -gather-v2-to-func input.mlir | \
mlir-opt -convert-gpu-to-nvvm -convert-func-to-llvm | \
mlir-translate --nvvm-to-ptx > kernel.ptx
```
The resulting PTX can be executed on NVIDIA GPUs.

### 5. Example Usage
Input TensorFlow:
```python
import tensorflow as tf
params0 = tf.random.uniform((10, 20))
params1 = tf.random.uniform((10, 20))
indices0 = tf.constant([0, 2, 4, 6, 8])
indices1 = tf.constant([1, 3, 5, 7, 9])
axis = tf.constant(0)
out0 = tf.gather(params0, indices0, axis=0)
out1 = tf.gather(params1, indices1, axis=0)
```
The fused function computes out0 and out1 in parallel on the GPU.

### 6. Optimization Opportunities
- Grid Scaling: For large num_pairs, split into multiple grid blocks and adjust block_x to a fixed size (e.g., 256).
- Shared Memory: Cache params or indices in #gpu.address_space<shared> for faster access within each block.
- Coalescing: Ensure memory accesses are coalesced (e.g., adjust thread mapping for axis = 1).
- Higher Ranks: Extend axis handling with a loop or switch for tensors with more than 2 dimensions.

## Benefits
- Scalability: Handles any number of params/indices pairs with the same shape.
- Efficiency: Single kernel reduces launch overhead and maximizes GPU utilization.
- Generality: Works with dynamic shapes and arbitrary axis values.
- Let me know if you’d like further refinements (e.g., higher-rank tensors or specific optimizations)!