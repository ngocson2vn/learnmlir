# kernel_creator.cc
https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/compiler/mlir/tools/kernel_gen/kernel_creator.cc

## Lowering Pipeline
TF dialect -> Loops -> GPU dialect -> NVVM dialect -> LLVM dialect -> LLVM IR -> PTX


```mlir
// -----// IR Dump Before LinalgLowerToParallelLoops (convert-linalg-to-parallel-loops) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<?xi32>)
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<?x2xi32>) outs(%alloc : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc : memref<?xi32>) outs(%alloc_0 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg1[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_2 : memref<?xi32>)
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<?x2xi32>) outs(%alloc_2 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_2 : memref<?xi32>) outs(%alloc_3 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg3[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_5 : memref<?xi32>)
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : memref<?x2xi32>) outs(%alloc_5 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_5 : memref<?xi32>) outs(%alloc_6 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg5[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After LinalgLowerToParallelLoops (convert-linalg-to-parallel-loops) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    memref.store %c0_i32, %alloc[%arg6] : memref<?xi32>
    scf.yield
  }
  %dim_0 = memref.dim %arg0, %c0 : memref<?x2xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_0) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_1 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_1[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_2 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_3 = memref.alloc(%dim_2) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_2) step (%c1) {
    memref.store %c0_i32, %alloc_3[%arg6] : memref<?xi32>
    scf.yield
  }
  %dim_4 = memref.dim %arg2, %c0 : memref<?x2xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_3[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_3[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_5 = memref.alloc(%dim_2) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_2, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_3[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_5[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_6 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_7 = memref.alloc(%dim_6) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_6) step (%c1) {
    memref.store %c0_i32, %alloc_7[%arg6] : memref<?xi32>
    scf.yield
  }
  %dim_8 = memref.dim %arg4, %c0 : memref<?x2xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_8) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_7[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_7[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_9 = memref.alloc(%dim_6) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_6, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_7[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_9[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_1, %alloc_5, %alloc_9 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}
```