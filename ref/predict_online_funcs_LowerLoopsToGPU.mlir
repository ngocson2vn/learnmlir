// -----// IR Dump Before GpuMapParallelLoopsPass (gpu-map-parallel-loops) //----- //
func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  %1 = arith.maxui %dim, %0 : index
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %2 = arith.maxui %1, %dim_1 : index
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %3 = arith.muli %dim_1, %c4 : index
  %4 = arith.maxui %2, %3 : index
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %5 = arith.maxui %4, %dim_4 : index
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %6 = arith.muli %dim_4, %c4 : index
  %7 = arith.maxui %5, %6 : index
  scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
    %8 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%14] : memref<?xi32>
      scf.yield
    }
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc[%14] : memref<?xi32>
      }
      scf.yield
    }
    %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
      scf.yield
    }
    %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
      scf.yield
    }
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_2[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_2[%14] : memref<?xi32>
      }
      scf.yield
    }
    %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%3]
    scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_2[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
      scf.yield
    }
    %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
      scf.yield
    }
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_5[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_5[%14] : memref<?xi32>
      }
      scf.yield
    }
    %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%6]
    scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_5[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After GpuMapParallelLoopsPass (gpu-map-parallel-loops) //----- //
func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  %1 = arith.maxui %dim, %0 : index
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %2 = arith.maxui %1, %dim_1 : index
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %3 = arith.muli %dim_1, %c4 : index
  %4 = arith.maxui %2, %3 : index
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %5 = arith.maxui %4, %dim_4 : index
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %6 = arith.muli %dim_4, %c4 : index
  %7 = arith.maxui %5, %6 : index
  scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
    %8 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_2[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_2[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%3]
    scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_2[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_5[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_5[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%6]
    scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_5[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before ExpandOps (memref-expand) //----- //
func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  %1 = arith.maxui %dim, %0 : index
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %2 = arith.maxui %1, %dim_1 : index
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %3 = arith.muli %dim_1, %c4 : index
  %4 = arith.maxui %2, %3 : index
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %5 = arith.maxui %4, %dim_4 : index
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %6 = arith.muli %dim_4, %c4 : index
  %7 = arith.maxui %5, %6 : index
  scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
    %8 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_2[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_2[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%3]
    scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_2[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_5[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_5[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%6]
    scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_5[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After ExpandOps (memref-expand) //----- //
func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  %1 = arith.maxui %dim, %0 : index
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %2 = arith.maxui %1, %dim_1 : index
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %3 = arith.muli %dim_1, %c4 : index
  %4 = arith.maxui %2, %3 : index
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %5 = arith.maxui %4, %dim_4 : index
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %6 = arith.muli %dim_4, %c4 : index
  %7 = arith.maxui %5, %6 : index
  scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
    %8 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_2[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_2[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%3]
    scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_2[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_5[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_5[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%6]
    scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_5[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
    %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
    %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
    %0 = arith.muli %dim, %c4 : index
    %1 = arith.maxui %dim, %0 : index
    %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
    %2 = arith.maxui %1, %dim_1 : index
    %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
    %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
    %3 = arith.muli %dim_1, %c4 : index
    %4 = arith.maxui %2, %3 : index
    %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
    %5 = arith.maxui %4, %dim_4 : index
    %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
    %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
    %6 = arith.muli %dim_4, %c4 : index
    %7 = arith.maxui %5, %6 : index
    scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
      %8 = affine.min #map(%arg6)[%dim]
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %9 = affine.min #map(%arg6)[%0]
      scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %10 = affine.min #map(%arg6)[%dim_1]
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_2[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_2[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %11 = affine.min #map(%arg6)[%3]
      scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_2[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %12 = affine.min #map(%arg6)[%dim_4]
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_5[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_5[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %13 = affine.min #map(%arg6)[%6]
      scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_5[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
    %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
    %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
    %0 = arith.muli %dim, %c4 : index
    %1 = arith.maxui %dim, %0 : index
    %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
    %2 = arith.maxui %1, %dim_1 : index
    %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
    %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
    %3 = arith.muli %dim_1, %c4 : index
    %4 = arith.maxui %2, %3 : index
    %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
    %5 = arith.maxui %4, %dim_4 : index
    %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
    %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
    %6 = arith.muli %dim_4, %c4 : index
    %7 = arith.maxui %5, %6 : index
    scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
      %8 = affine.min #map(%arg6)[%dim]
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %9 = affine.min #map(%arg6)[%0]
      scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %10 = affine.min #map(%arg6)[%dim_1]
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_2[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_2[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %11 = affine.min #map(%arg6)[%3]
      scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_2[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %12 = affine.min #map(%arg6)[%dim_4]
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_5[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_5[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %13 = affine.min #map(%arg6)[%6]
      scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_5[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump Before ShapeToDescriptorsPass (shape-to-descriptors) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
    %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
    %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
    %0 = arith.muli %dim, %c4 : index
    %1 = arith.maxui %dim, %0 : index
    %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
    %2 = arith.maxui %1, %dim_1 : index
    %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
    %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
    %3 = arith.muli %dim_1, %c4 : index
    %4 = arith.maxui %2, %3 : index
    %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
    %5 = arith.maxui %4, %dim_4 : index
    %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
    %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
    %6 = arith.muli %dim_4, %c4 : index
    %7 = arith.maxui %5, %6 : index
    scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
      %8 = affine.min #map(%arg6)[%dim]
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %9 = affine.min #map(%arg6)[%0]
      scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %10 = affine.min #map(%arg6)[%dim_1]
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_2[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_2[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %11 = affine.min #map(%arg6)[%3]
      scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_2[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %12 = affine.min #map(%arg6)[%dim_4]
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_5[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_5[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %13 = affine.min #map(%arg6)[%6]
      scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_5[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump After ShapeToDescriptorsPass (shape-to-descriptors) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
    %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
    %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
    %0 = arith.muli %dim, %c4 : index
    %1 = arith.maxui %dim, %0 : index
    %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
    %2 = arith.maxui %1, %dim_1 : index
    %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
    %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
    %3 = arith.muli %dim_1, %c4 : index
    %4 = arith.maxui %2, %3 : index
    %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
    %5 = arith.maxui %4, %dim_4 : index
    %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
    %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
    %6 = arith.muli %dim_4, %c4 : index
    %7 = arith.maxui %5, %6 : index
    scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
      %8 = affine.min #map(%arg6)[%dim]
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %9 = affine.min #map(%arg6)[%0]
      scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %10 = affine.min #map(%arg6)[%dim_1]
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_2[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_2[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %11 = affine.min #map(%arg6)[%3]
      scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_2[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %12 = affine.min #map(%arg6)[%dim_4]
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_5[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_5[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %13 = affine.min #map(%arg6)[%6]
      scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_5[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
    %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
    %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
    %0 = arith.muli %dim, %c4 : index
    %1 = arith.maxui %dim, %0 : index
    %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
    %2 = arith.maxui %1, %dim_1 : index
    %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
    %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
    %3 = arith.muli %dim_1, %c4 : index
    %4 = arith.maxui %2, %3 : index
    %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
    %5 = arith.maxui %4, %dim_4 : index
    %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
    %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
    %6 = arith.muli %dim_4, %c4 : index
    %7 = arith.maxui %5, %6 : index
    scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
      %8 = affine.min #map(%arg6)[%dim]
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %9 = affine.min #map(%arg6)[%0]
      scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %10 = affine.min #map(%arg6)[%dim_1]
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_2[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_2[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %11 = affine.min #map(%arg6)[%3]
      scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_2[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %12 = affine.min #map(%arg6)[%dim_4]
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_5[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_5[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %13 = affine.min #map(%arg6)[%6]
      scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_5[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
    %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
    %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
    %0 = arith.muli %dim, %c4 : index
    %1 = arith.maxui %dim, %0 : index
    %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
    %2 = arith.maxui %1, %dim_1 : index
    %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
    %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
    %3 = arith.muli %dim_1, %c4 : index
    %4 = arith.maxui %2, %3 : index
    %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
    %5 = arith.maxui %4, %dim_4 : index
    %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
    %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
    %6 = arith.muli %dim_4, %c4 : index
    %7 = arith.maxui %5, %6 : index
    scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
      %8 = affine.min #map(%arg6)[%dim]
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %9 = affine.min #map(%arg6)[%0]
      scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %10 = affine.min #map(%arg6)[%dim_1]
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_2[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_2[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %11 = affine.min #map(%arg6)[%3]
      scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_2[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %12 = affine.min #map(%arg6)[%dim_4]
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_5[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_5[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %13 = affine.min #map(%arg6)[%6]
      scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_5[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump Before CSE (cse) //----- //
func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  %1 = arith.maxui %dim, %0 : index
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %2 = arith.maxui %1, %dim_1 : index
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %3 = arith.muli %dim_1, %c4 : index
  %4 = arith.maxui %2, %3 : index
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %5 = arith.maxui %4, %dim_4 : index
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %6 = arith.muli %dim_4, %c4 : index
  %7 = arith.maxui %5, %6 : index
  scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
    %8 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_2[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_2[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%3]
    scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_2[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_5[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_5[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%6]
    scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_5[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  %1 = arith.maxui %dim, %0 : index
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %2 = arith.maxui %1, %dim_1 : index
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %3 = arith.muli %dim_1, %c4 : index
  %4 = arith.maxui %2, %3 : index
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %5 = arith.maxui %4, %dim_4 : index
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %6 = arith.muli %dim_4, %c4 : index
  %7 = arith.maxui %5, %6 : index
  scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
    %8 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_2[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_2[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%3]
    scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_2[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
        %16 = memref.load %alloc_5[%14] : memref<?xi32>
        %17 = arith.addi %16, %15 : i32
        memref.store %17, %alloc_5[%14] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%6]
    scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
      %14 = arith.addi %arg7, %arg6 : index
      %15 = arith.remsi %14, %c4 : index
      %16 = arith.divsi %14, %c4 : index
      %17 = memref.load %alloc_5[%16] : memref<?xi32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
      memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before EmbedTFFrameworkPass (embed-tf-framework) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
    %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
    %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
    %0 = arith.muli %dim, %c4 : index
    %1 = arith.maxui %dim, %0 : index
    %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
    %2 = arith.maxui %1, %dim_1 : index
    %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
    %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
    %3 = arith.muli %dim_1, %c4 : index
    %4 = arith.maxui %2, %3 : index
    %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
    %5 = arith.maxui %4, %dim_4 : index
    %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
    %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
    %6 = arith.muli %dim_4, %c4 : index
    %7 = arith.maxui %5, %6 : index
    scf.parallel (%arg6) = (%c0) to (%7) step (%c512) {
      %8 = affine.min #map(%arg6)[%dim]
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%8) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg0[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %9 = affine.min #map(%arg6)[%0]
      scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg1[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_0[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %10 = affine.min #map(%arg6)[%dim_1]
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_2[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg2[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_2[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_2[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %11 = affine.min #map(%arg6)[%3]
      scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_2[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg3[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_3[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %12 = affine.min #map(%arg6)[%dim_4]
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        memref.store %c0_i32, %alloc_5[%14] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        scf.for %arg8 = %c0 to %c2 step %c1 {
          %15 = memref.load %arg4[%14, %arg8] : memref<?x2xi32>
          %16 = memref.load %alloc_5[%14] : memref<?xi32>
          %17 = arith.addi %16, %15 : i32
          memref.store %17, %alloc_5[%14] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %13 = affine.min #map(%arg6)[%6]
      scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
        %14 = arith.addi %arg7, %arg6 : index
        %15 = arith.remsi %14, %c4 : index
        %16 = arith.divsi %14, %c4 : index
        %17 = memref.load %alloc_5[%16] : memref<?xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = memref.load %arg5[%18, %15] : memref<?x4xf32>
        memref.store %19, %alloc_6[%16, %15] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump After EmbedTFFrameworkPass (embed-tf-framework) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
      %20 = affine.min #map(%arg7)[%dim]
      scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %0[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %0[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %0[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %21 = affine.min #map(%arg7)[%4]
      scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %0[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
        memref.store %31, %2[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %22 = affine.min #map(%arg7)[%dim_0]
      scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %7[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %7[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %7[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %23 = affine.min #map(%arg7)[%11]
      scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %7[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
        memref.store %31, %9[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %24 = affine.min #map(%arg7)[%dim_1]
      scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %14[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %14[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %14[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %25 = affine.min #map(%arg7)[%18]
      scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %14[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
        memref.store %31, %16[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump Before AFCheckExternalCallResult (af-check-external-call-result) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
    %20 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim]
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %0[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %0[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %0[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %21 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%4]
    scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %0[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
      memref.store %31, %2[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %22 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_0]
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %7[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %7[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %7[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %23 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%11]
    scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %7[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
      memref.store %31, %9[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %24 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_1]
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %14[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %14[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %14[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %25 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%18]
    scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %14[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
      memref.store %31, %16[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After AFCheckExternalCallResult (af-check-external-call-result) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
    %20 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim]
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %0[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %0[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %0[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %21 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%4]
    scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %0[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
      memref.store %31, %2[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %22 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_0]
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %7[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %7[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %7[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %23 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%11]
    scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %7[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
      memref.store %31, %9[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %24 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_1]
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %14[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %14[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %14[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %25 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%18]
    scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %14[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
      memref.store %31, %16[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before FinalBufferizePass (final-bufferize) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
      %20 = affine.min #map(%arg7)[%dim]
      scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %0[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %0[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %0[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %21 = affine.min #map(%arg7)[%4]
      scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %0[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
        memref.store %31, %2[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %22 = affine.min #map(%arg7)[%dim_0]
      scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %7[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %7[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %7[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %23 = affine.min #map(%arg7)[%11]
      scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %7[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
        memref.store %31, %9[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %24 = affine.min #map(%arg7)[%dim_1]
      scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %14[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %14[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %14[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %25 = affine.min #map(%arg7)[%18]
      scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %14[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
        memref.store %31, %16[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump After FinalBufferizePass (final-bufferize) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
      %20 = affine.min #map(%arg7)[%dim]
      scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %0[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %0[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %0[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %21 = affine.min #map(%arg7)[%4]
      scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %0[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
        memref.store %31, %2[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %22 = affine.min #map(%arg7)[%dim_0]
      scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %7[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %7[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %7[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %23 = affine.min #map(%arg7)[%11]
      scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %7[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
        memref.store %31, %9[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %24 = affine.min #map(%arg7)[%dim_1]
      scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %14[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %14[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %14[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %25 = affine.min #map(%arg7)[%18]
      scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %14[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
        memref.store %31, %16[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump Before BufferHoisting (buffer-hoisting) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
    %20 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim]
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %0[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %0[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %0[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %21 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%4]
    scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %0[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
      memref.store %31, %2[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %22 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_0]
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %7[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %7[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %7[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %23 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%11]
    scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %7[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
      memref.store %31, %9[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %24 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_1]
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %14[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %14[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %14[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %25 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%18]
    scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %14[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
      memref.store %31, %16[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After BufferHoisting (buffer-hoisting) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
    %20 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim]
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %0[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %0[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %0[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %21 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%4]
    scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %0[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
      memref.store %31, %2[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %22 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_0]
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %7[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %7[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %7[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %23 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%11]
    scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %7[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
      memref.store %31, %9[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %24 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_1]
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %14[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %14[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %14[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %25 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%18]
    scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %14[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
      memref.store %31, %16[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before BufferDeallocation (buffer-deallocation) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
    %20 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim]
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %0[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %0[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %0[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %21 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%4]
    scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %0[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
      memref.store %31, %2[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %22 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_0]
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %7[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %7[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %7[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %23 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%11]
    scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %7[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
      memref.store %31, %9[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %24 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_1]
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %14[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %14[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %14[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %25 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%18]
    scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %14[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
      memref.store %31, %16[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After BufferDeallocation (buffer-deallocation) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
    %20 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim]
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %0[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %0[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %0[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %21 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%4]
    scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %0[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
      memref.store %31, %2[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %22 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_0]
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %7[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %7[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %7[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %23 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%11]
    scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %7[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
      memref.store %31, %9[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %24 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_1]
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %14[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %14[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %14[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %25 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%18]
    scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %14[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
      memref.store %31, %16[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
      %20 = affine.min #map(%arg7)[%dim]
      scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %0[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %0[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %0[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %21 = affine.min #map(%arg7)[%4]
      scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %0[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
        memref.store %31, %2[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %22 = affine.min #map(%arg7)[%dim_0]
      scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %7[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %7[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %7[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %23 = affine.min #map(%arg7)[%11]
      scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %7[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
        memref.store %31, %9[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %24 = affine.min #map(%arg7)[%dim_1]
      scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %14[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %14[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %14[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %25 = affine.min #map(%arg7)[%18]
      scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %14[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
        memref.store %31, %16[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
      %20 = affine.min #map(%arg7)[%dim]
      scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %0[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %0[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %0[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %21 = affine.min #map(%arg7)[%4]
      scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %0[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
        memref.store %31, %2[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %22 = affine.min #map(%arg7)[%dim_0]
      scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %7[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %7[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %7[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %23 = affine.min #map(%arg7)[%11]
      scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %7[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
        memref.store %31, %9[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %24 = affine.min #map(%arg7)[%dim_1]
      scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        memref.store %c0_i32, %14[%26] : memref<?xi32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        scf.for %arg9 = %c0 to %c2 step %c1 {
          %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
          %28 = memref.load %14[%26] : memref<?xi32>
          %29 = arith.addi %28, %27 : i32
          memref.store %29, %14[%26] : memref<?xi32>
        }
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      %25 = affine.min #map(%arg7)[%18]
      scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
        %26 = arith.addi %arg8, %arg7 : index
        %27 = arith.remsi %26, %c4 : index
        %28 = arith.divsi %26, %c4 : index
        %29 = memref.load %14[%28] : memref<?xi32>
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
        memref.store %31, %16[%28, %27] : memref<?x4xf32>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump Before CustomConvertParallelLoopsToGpu (custom-convert-parallel-loops-to-gpu) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  scf.parallel (%arg7) = (%c0) to (%19) step (%c512) {
    %20 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim]
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %0[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%20) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg1[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %0[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %0[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %21 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%4]
    scf.parallel (%arg8) = (%c0) to (%21) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %0[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg2[%30, %27] : memref<?x4xf32>
      memref.store %31, %2[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %22 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_0]
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %7[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%22) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg3[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %7[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %7[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %23 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%11]
    scf.parallel (%arg8) = (%c0) to (%23) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %7[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg4[%30, %27] : memref<?x4xf32>
      memref.store %31, %9[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %24 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%dim_1]
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      memref.store %c0_i32, %14[%26] : memref<?xi32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.parallel (%arg8) = (%c0) to (%24) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      scf.for %arg9 = %c0 to %c2 step %c1 {
        %27 = memref.load %arg5[%26, %arg9] : memref<?x2xi32>
        %28 = memref.load %14[%26] : memref<?xi32>
        %29 = arith.addi %28, %27 : i32
        memref.store %29, %14[%26] : memref<?xi32>
      }
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    %25 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg7)[%18]
    scf.parallel (%arg8) = (%c0) to (%25) step (%c1) {
      %26 = arith.addi %arg8, %arg7 : index
      %27 = arith.remsi %26, %c4 : index
      %28 = arith.divsi %26, %c4 : index
      %29 = memref.load %14[%28] : memref<?xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %arg6[%30, %27] : memref<?x4xf32>
      memref.store %31, %16[%28, %27] : memref<?x4xf32>
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    scf.yield
  } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After CustomConvertParallelLoopsToGpu (custom-convert-parallel-loops-to-gpu) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c7 = arith.constant 7 : index
  %c14 = arith.constant 14 : index
  %c21 = arith.constant 21 : index
  %c28 = arith.constant 28 : index
  %c35 = arith.constant 35 : index
  %c42 = arith.constant 42 : index
  %c49 = arith.constant 49 : index
  %c56 = arith.constant 56 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%21)[%c0, %c512]
  gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
    %23 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %24 = arith.remsi %arg7, %arg13 : index
    %25 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%24)[%c512, %c0]
    %26 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%25)[%dim]
    %27 = arith.cmpi slt, %23, %26 : index
    scf.if %27 {
      %76 = arith.addi %23, %25 : index
      memref.store %c0_i32, %0[%76] : memref<?xi32>
    }
    %28 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %29 = arith.addi %arg7, %c7 : index
    %30 = arith.remsi %29, %arg13 : index
    %31 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%30)[%c512, %c0]
    %32 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%31)[%dim]
    %33 = arith.cmpi slt, %28, %32 : index
    scf.if %33 {
      %76 = arith.addi %28, %31 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %77 = memref.load %arg1[%76, %arg19] : memref<?x2xi32>
        %78 = memref.load %0[%76] : memref<?xi32>
        %79 = arith.addi %78, %77 : i32
        memref.store %79, %0[%76] : memref<?xi32>
      }
    }
    %34 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %35 = arith.addi %arg7, %c14 : index
    %36 = arith.remsi %35, %arg13 : index
    %37 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%36)[%c512, %c0]
    %38 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%37)[%4]
    %39 = arith.cmpi slt, %34, %38 : index
    scf.if %39 {
      %76 = arith.addi %34, %37 : index
      %77 = arith.remsi %76, %c4 : index
      %78 = arith.divsi %76, %c4 : index
      %79 = memref.load %0[%78] : memref<?xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = memref.load %arg2[%80, %77] : memref<?x4xf32>
      memref.store %81, %2[%78, %77] : memref<?x4xf32>
    }
    %40 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %41 = arith.addi %arg7, %c21 : index
    %42 = arith.remsi %41, %arg13 : index
    %43 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%42)[%c512, %c0]
    %44 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%43)[%dim_0]
    %45 = arith.cmpi slt, %40, %44 : index
    scf.if %45 {
      %76 = arith.addi %40, %43 : index
      memref.store %c0_i32, %7[%76] : memref<?xi32>
    }
    %46 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %47 = arith.addi %arg7, %c28 : index
    %48 = arith.remsi %47, %arg13 : index
    %49 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%48)[%c512, %c0]
    %50 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%49)[%dim_0]
    %51 = arith.cmpi slt, %46, %50 : index
    scf.if %51 {
      %76 = arith.addi %46, %49 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %77 = memref.load %arg3[%76, %arg19] : memref<?x2xi32>
        %78 = memref.load %7[%76] : memref<?xi32>
        %79 = arith.addi %78, %77 : i32
        memref.store %79, %7[%76] : memref<?xi32>
      }
    }
    %52 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %53 = arith.addi %arg7, %c35 : index
    %54 = arith.remsi %53, %arg13 : index
    %55 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%54)[%c512, %c0]
    %56 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%55)[%11]
    %57 = arith.cmpi slt, %52, %56 : index
    scf.if %57 {
      %76 = arith.addi %52, %55 : index
      %77 = arith.remsi %76, %c4 : index
      %78 = arith.divsi %76, %c4 : index
      %79 = memref.load %7[%78] : memref<?xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = memref.load %arg4[%80, %77] : memref<?x4xf32>
      memref.store %81, %9[%78, %77] : memref<?x4xf32>
    }
    %58 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %59 = arith.addi %arg7, %c42 : index
    %60 = arith.remsi %59, %arg13 : index
    %61 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%60)[%c512, %c0]
    %62 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%61)[%dim_1]
    %63 = arith.cmpi slt, %58, %62 : index
    scf.if %63 {
      %76 = arith.addi %58, %61 : index
      memref.store %c0_i32, %14[%76] : memref<?xi32>
    }
    %64 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %65 = arith.addi %arg7, %c49 : index
    %66 = arith.remsi %65, %arg13 : index
    %67 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%66)[%c512, %c0]
    %68 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%67)[%dim_1]
    %69 = arith.cmpi slt, %64, %68 : index
    scf.if %69 {
      %76 = arith.addi %64, %67 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %77 = memref.load %arg5[%76, %arg19] : memref<?x2xi32>
        %78 = memref.load %14[%76] : memref<?xi32>
        %79 = arith.addi %78, %77 : i32
        memref.store %79, %14[%76] : memref<?xi32>
      }
    }
    %70 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %71 = arith.addi %arg7, %c56 : index
    %72 = arith.remsi %71, %arg13 : index
    %73 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%72)[%c512, %c0]
    %74 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%73)[%18]
    %75 = arith.cmpi slt, %70, %74 : index
    scf.if %75 {
      %76 = arith.addi %70, %73 : index
      %77 = arith.remsi %76, %c4 : index
      %78 = arith.divsi %76, %c4 : index
      %79 = memref.load %14[%78] : memref<?xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = memref.load %arg6[%80, %77] : memref<?x4xf32>
      memref.store %81, %16[%78, %77] : memref<?x4xf32>
    }
    gpu.terminator
  } {SCFToGPU_visited}
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c7 = arith.constant 7 : index
  %c14 = arith.constant 14 : index
  %c21 = arith.constant 21 : index
  %c28 = arith.constant 28 : index
  %c35 = arith.constant 35 : index
  %c42 = arith.constant 42 : index
  %c49 = arith.constant 49 : index
  %c56 = arith.constant 56 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%21)[%c0, %c512]
  gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
    %23 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %24 = arith.remsi %arg7, %arg13 : index
    %25 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%24)[%c512, %c0]
    %26 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%25)[%dim]
    %27 = arith.cmpi slt, %23, %26 : index
    scf.if %27 {
      %76 = arith.addi %23, %25 : index
      memref.store %c0_i32, %0[%76] : memref<?xi32>
    }
    %28 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %29 = arith.addi %arg7, %c7 : index
    %30 = arith.remsi %29, %arg13 : index
    %31 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%30)[%c512, %c0]
    %32 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%31)[%dim]
    %33 = arith.cmpi slt, %28, %32 : index
    scf.if %33 {
      %76 = arith.addi %28, %31 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %77 = memref.load %arg1[%76, %arg19] : memref<?x2xi32>
        %78 = memref.load %0[%76] : memref<?xi32>
        %79 = arith.addi %78, %77 : i32
        memref.store %79, %0[%76] : memref<?xi32>
      }
    }
    %34 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %35 = arith.addi %arg7, %c14 : index
    %36 = arith.remsi %35, %arg13 : index
    %37 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%36)[%c512, %c0]
    %38 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%37)[%4]
    %39 = arith.cmpi slt, %34, %38 : index
    scf.if %39 {
      %76 = arith.addi %34, %37 : index
      %77 = arith.remsi %76, %c4 : index
      %78 = arith.divsi %76, %c4 : index
      %79 = memref.load %0[%78] : memref<?xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = memref.load %arg2[%80, %77] : memref<?x4xf32>
      memref.store %81, %2[%78, %77] : memref<?x4xf32>
    }
    %40 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %41 = arith.addi %arg7, %c21 : index
    %42 = arith.remsi %41, %arg13 : index
    %43 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%42)[%c512, %c0]
    %44 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%43)[%dim_0]
    %45 = arith.cmpi slt, %40, %44 : index
    scf.if %45 {
      %76 = arith.addi %40, %43 : index
      memref.store %c0_i32, %7[%76] : memref<?xi32>
    }
    %46 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %47 = arith.addi %arg7, %c28 : index
    %48 = arith.remsi %47, %arg13 : index
    %49 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%48)[%c512, %c0]
    %50 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%49)[%dim_0]
    %51 = arith.cmpi slt, %46, %50 : index
    scf.if %51 {
      %76 = arith.addi %46, %49 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %77 = memref.load %arg3[%76, %arg19] : memref<?x2xi32>
        %78 = memref.load %7[%76] : memref<?xi32>
        %79 = arith.addi %78, %77 : i32
        memref.store %79, %7[%76] : memref<?xi32>
      }
    }
    %52 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %53 = arith.addi %arg7, %c35 : index
    %54 = arith.remsi %53, %arg13 : index
    %55 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%54)[%c512, %c0]
    %56 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%55)[%11]
    %57 = arith.cmpi slt, %52, %56 : index
    scf.if %57 {
      %76 = arith.addi %52, %55 : index
      %77 = arith.remsi %76, %c4 : index
      %78 = arith.divsi %76, %c4 : index
      %79 = memref.load %7[%78] : memref<?xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = memref.load %arg4[%80, %77] : memref<?x4xf32>
      memref.store %81, %9[%78, %77] : memref<?x4xf32>
    }
    %58 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %59 = arith.addi %arg7, %c42 : index
    %60 = arith.remsi %59, %arg13 : index
    %61 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%60)[%c512, %c0]
    %62 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%61)[%dim_1]
    %63 = arith.cmpi slt, %58, %62 : index
    scf.if %63 {
      %76 = arith.addi %58, %61 : index
      memref.store %c0_i32, %14[%76] : memref<?xi32>
    }
    %64 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %65 = arith.addi %arg7, %c49 : index
    %66 = arith.remsi %65, %arg13 : index
    %67 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%66)[%c512, %c0]
    %68 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%67)[%dim_1]
    %69 = arith.cmpi slt, %64, %68 : index
    scf.if %69 {
      %76 = arith.addi %64, %67 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %77 = memref.load %arg5[%76, %arg19] : memref<?x2xi32>
        %78 = memref.load %14[%76] : memref<?xi32>
        %79 = arith.addi %78, %77 : i32
        memref.store %79, %14[%76] : memref<?xi32>
      }
    }
    %70 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg10)[%c1, %c0]
    %71 = arith.addi %arg7, %c56 : index
    %72 = arith.remsi %71, %arg13 : index
    %73 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%72)[%c512, %c0]
    %74 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%73)[%18]
    %75 = arith.cmpi slt, %70, %74 : index
    scf.if %75 {
      %76 = arith.addi %70, %73 : index
      %77 = arith.remsi %76, %c4 : index
      %78 = arith.divsi %76, %c4 : index
      %79 = memref.load %14[%78] : memref<?xi32>
      %80 = arith.index_cast %79 : i32 to index
      %81 = memref.load %arg6[%80, %77] : memref<?x4xf32>
      memref.store %81, %16[%78, %77] : memref<?x4xf32>
    }
    gpu.terminator
  } {SCFToGPU_visited}
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c7 = arith.constant 7 : index
  %c14 = arith.constant 14 : index
  %c21 = arith.constant 21 : index
  %c28 = arith.constant 28 : index
  %c35 = arith.constant 35 : index
  %c42 = arith.constant 42 : index
  %c49 = arith.constant 49 : index
  %c56 = arith.constant 56 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = affine.apply affine_map<()[s0] -> (s0 ceildiv 512)>()[%21]
  gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
    %23 = arith.remsi %arg7, %arg13 : index
    %24 = affine.apply affine_map<(d0) -> (d0 * 512)>(%23)
    %25 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%23)[%dim]
    %26 = arith.cmpi slt, %arg10, %25 : index
    scf.if %26 {
      %67 = arith.addi %arg10, %24 : index
      memref.store %c0_i32, %0[%67] : memref<?xi32>
    }
    %27 = arith.addi %arg7, %c7 : index
    %28 = arith.remsi %27, %arg13 : index
    %29 = affine.apply affine_map<(d0) -> (d0 * 512)>(%28)
    %30 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%28)[%dim]
    %31 = arith.cmpi slt, %arg10, %30 : index
    scf.if %31 {
      %67 = arith.addi %arg10, %29 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg1[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %0[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %0[%67] : memref<?xi32>
      }
    }
    %32 = arith.addi %arg7, %c14 : index
    %33 = arith.remsi %32, %arg13 : index
    %34 = affine.apply affine_map<(d0) -> (d0 * 512)>(%33)
    %35 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%33)[%4]
    %36 = arith.cmpi slt, %arg10, %35 : index
    scf.if %36 {
      %67 = arith.addi %arg10, %34 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %0[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg2[%71, %68] : memref<?x4xf32>
      memref.store %72, %2[%69, %68] : memref<?x4xf32>
    }
    %37 = arith.addi %arg7, %c21 : index
    %38 = arith.remsi %37, %arg13 : index
    %39 = affine.apply affine_map<(d0) -> (d0 * 512)>(%38)
    %40 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%38)[%dim_0]
    %41 = arith.cmpi slt, %arg10, %40 : index
    scf.if %41 {
      %67 = arith.addi %arg10, %39 : index
      memref.store %c0_i32, %7[%67] : memref<?xi32>
    }
    %42 = arith.addi %arg7, %c28 : index
    %43 = arith.remsi %42, %arg13 : index
    %44 = affine.apply affine_map<(d0) -> (d0 * 512)>(%43)
    %45 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%43)[%dim_0]
    %46 = arith.cmpi slt, %arg10, %45 : index
    scf.if %46 {
      %67 = arith.addi %arg10, %44 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg3[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %7[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %7[%67] : memref<?xi32>
      }
    }
    %47 = arith.addi %arg7, %c35 : index
    %48 = arith.remsi %47, %arg13 : index
    %49 = affine.apply affine_map<(d0) -> (d0 * 512)>(%48)
    %50 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%48)[%11]
    %51 = arith.cmpi slt, %arg10, %50 : index
    scf.if %51 {
      %67 = arith.addi %arg10, %49 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %7[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg4[%71, %68] : memref<?x4xf32>
      memref.store %72, %9[%69, %68] : memref<?x4xf32>
    }
    %52 = arith.addi %arg7, %c42 : index
    %53 = arith.remsi %52, %arg13 : index
    %54 = affine.apply affine_map<(d0) -> (d0 * 512)>(%53)
    %55 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%53)[%dim_1]
    %56 = arith.cmpi slt, %arg10, %55 : index
    scf.if %56 {
      %67 = arith.addi %arg10, %54 : index
      memref.store %c0_i32, %14[%67] : memref<?xi32>
    }
    %57 = arith.addi %arg7, %c49 : index
    %58 = arith.remsi %57, %arg13 : index
    %59 = affine.apply affine_map<(d0) -> (d0 * 512)>(%58)
    %60 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%58)[%dim_1]
    %61 = arith.cmpi slt, %arg10, %60 : index
    scf.if %61 {
      %67 = arith.addi %arg10, %59 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg5[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %14[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %14[%67] : memref<?xi32>
      }
    }
    %62 = arith.addi %arg7, %c56 : index
    %63 = arith.remsi %62, %arg13 : index
    %64 = affine.apply affine_map<(d0) -> (d0 * 512)>(%63)
    %65 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%63)[%18]
    %66 = arith.cmpi slt, %arg10, %65 : index
    scf.if %66 {
      %67 = arith.addi %arg10, %64 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %14[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg6[%71, %68] : memref<?x4xf32>
      memref.store %72, %16[%69, %68] : memref<?x4xf32>
    }
    gpu.terminator
  } {SCFToGPU_visited}
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c7 = arith.constant 7 : index
  %c14 = arith.constant 14 : index
  %c21 = arith.constant 21 : index
  %c28 = arith.constant 28 : index
  %c35 = arith.constant 35 : index
  %c42 = arith.constant 42 : index
  %c49 = arith.constant 49 : index
  %c56 = arith.constant 56 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = affine.apply affine_map<()[s0] -> (s0 ceildiv 512)>()[%21]
  gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
    %23 = arith.remsi %arg7, %arg13 : index
    %24 = affine.apply affine_map<(d0) -> (d0 * 512)>(%23)
    %25 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%23)[%dim]
    %26 = arith.cmpi slt, %arg10, %25 : index
    scf.if %26 {
      %67 = arith.addi %arg10, %24 : index
      memref.store %c0_i32, %0[%67] : memref<?xi32>
    }
    %27 = arith.addi %arg7, %c7 : index
    %28 = arith.remsi %27, %arg13 : index
    %29 = affine.apply affine_map<(d0) -> (d0 * 512)>(%28)
    %30 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%28)[%dim]
    %31 = arith.cmpi slt, %arg10, %30 : index
    scf.if %31 {
      %67 = arith.addi %arg10, %29 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg1[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %0[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %0[%67] : memref<?xi32>
      }
    }
    %32 = arith.addi %arg7, %c14 : index
    %33 = arith.remsi %32, %arg13 : index
    %34 = affine.apply affine_map<(d0) -> (d0 * 512)>(%33)
    %35 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%33)[%4]
    %36 = arith.cmpi slt, %arg10, %35 : index
    scf.if %36 {
      %67 = arith.addi %arg10, %34 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %0[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg2[%71, %68] : memref<?x4xf32>
      memref.store %72, %2[%69, %68] : memref<?x4xf32>
    }
    %37 = arith.addi %arg7, %c21 : index
    %38 = arith.remsi %37, %arg13 : index
    %39 = affine.apply affine_map<(d0) -> (d0 * 512)>(%38)
    %40 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%38)[%dim_0]
    %41 = arith.cmpi slt, %arg10, %40 : index
    scf.if %41 {
      %67 = arith.addi %arg10, %39 : index
      memref.store %c0_i32, %7[%67] : memref<?xi32>
    }
    %42 = arith.addi %arg7, %c28 : index
    %43 = arith.remsi %42, %arg13 : index
    %44 = affine.apply affine_map<(d0) -> (d0 * 512)>(%43)
    %45 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%43)[%dim_0]
    %46 = arith.cmpi slt, %arg10, %45 : index
    scf.if %46 {
      %67 = arith.addi %arg10, %44 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg3[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %7[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %7[%67] : memref<?xi32>
      }
    }
    %47 = arith.addi %arg7, %c35 : index
    %48 = arith.remsi %47, %arg13 : index
    %49 = affine.apply affine_map<(d0) -> (d0 * 512)>(%48)
    %50 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%48)[%11]
    %51 = arith.cmpi slt, %arg10, %50 : index
    scf.if %51 {
      %67 = arith.addi %arg10, %49 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %7[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg4[%71, %68] : memref<?x4xf32>
      memref.store %72, %9[%69, %68] : memref<?x4xf32>
    }
    %52 = arith.addi %arg7, %c42 : index
    %53 = arith.remsi %52, %arg13 : index
    %54 = affine.apply affine_map<(d0) -> (d0 * 512)>(%53)
    %55 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%53)[%dim_1]
    %56 = arith.cmpi slt, %arg10, %55 : index
    scf.if %56 {
      %67 = arith.addi %arg10, %54 : index
      memref.store %c0_i32, %14[%67] : memref<?xi32>
    }
    %57 = arith.addi %arg7, %c49 : index
    %58 = arith.remsi %57, %arg13 : index
    %59 = affine.apply affine_map<(d0) -> (d0 * 512)>(%58)
    %60 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%58)[%dim_1]
    %61 = arith.cmpi slt, %arg10, %60 : index
    scf.if %61 {
      %67 = arith.addi %arg10, %59 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg5[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %14[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %14[%67] : memref<?xi32>
      }
    }
    %62 = arith.addi %arg7, %c56 : index
    %63 = arith.remsi %62, %arg13 : index
    %64 = affine.apply affine_map<(d0) -> (d0 * 512)>(%63)
    %65 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%63)[%18]
    %66 = arith.cmpi slt, %arg10, %65 : index
    scf.if %66 {
      %67 = arith.addi %arg10, %64 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %14[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg6[%71, %68] : memref<?x4xf32>
      memref.store %72, %16[%69, %68] : memref<?x4xf32>
    }
    gpu.terminator
  } {SCFToGPU_visited}
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c7 = arith.constant 7 : index
  %c14 = arith.constant 14 : index
  %c21 = arith.constant 21 : index
  %c28 = arith.constant 28 : index
  %c35 = arith.constant 35 : index
  %c42 = arith.constant 42 : index
  %c49 = arith.constant 49 : index
  %c56 = arith.constant 56 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = affine.apply affine_map<()[s0] -> (s0 ceildiv 512)>()[%21]
  gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
    %23 = arith.remsi %arg7, %arg13 : index
    %24 = affine.apply affine_map<(d0) -> (d0 * 512)>(%23)
    %25 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%23)[%dim]
    %26 = arith.cmpi slt, %arg10, %25 : index
    scf.if %26 {
      %67 = arith.addi %arg10, %24 : index
      memref.store %c0_i32, %0[%67] : memref<?xi32>
    }
    %27 = arith.addi %arg7, %c7 : index
    %28 = arith.remsi %27, %arg13 : index
    %29 = affine.apply affine_map<(d0) -> (d0 * 512)>(%28)
    %30 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%28)[%dim]
    %31 = arith.cmpi slt, %arg10, %30 : index
    scf.if %31 {
      %67 = arith.addi %arg10, %29 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg1[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %0[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %0[%67] : memref<?xi32>
      }
    }
    %32 = arith.addi %arg7, %c14 : index
    %33 = arith.remsi %32, %arg13 : index
    %34 = affine.apply affine_map<(d0) -> (d0 * 512)>(%33)
    %35 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%33)[%4]
    %36 = arith.cmpi slt, %arg10, %35 : index
    scf.if %36 {
      %67 = arith.addi %arg10, %34 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %0[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg2[%71, %68] : memref<?x4xf32>
      memref.store %72, %2[%69, %68] : memref<?x4xf32>
    }
    %37 = arith.addi %arg7, %c21 : index
    %38 = arith.remsi %37, %arg13 : index
    %39 = affine.apply affine_map<(d0) -> (d0 * 512)>(%38)
    %40 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%38)[%dim_0]
    %41 = arith.cmpi slt, %arg10, %40 : index
    scf.if %41 {
      %67 = arith.addi %arg10, %39 : index
      memref.store %c0_i32, %7[%67] : memref<?xi32>
    }
    %42 = arith.addi %arg7, %c28 : index
    %43 = arith.remsi %42, %arg13 : index
    %44 = affine.apply affine_map<(d0) -> (d0 * 512)>(%43)
    %45 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%43)[%dim_0]
    %46 = arith.cmpi slt, %arg10, %45 : index
    scf.if %46 {
      %67 = arith.addi %arg10, %44 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg3[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %7[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %7[%67] : memref<?xi32>
      }
    }
    %47 = arith.addi %arg7, %c35 : index
    %48 = arith.remsi %47, %arg13 : index
    %49 = affine.apply affine_map<(d0) -> (d0 * 512)>(%48)
    %50 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%48)[%11]
    %51 = arith.cmpi slt, %arg10, %50 : index
    scf.if %51 {
      %67 = arith.addi %arg10, %49 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %7[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg4[%71, %68] : memref<?x4xf32>
      memref.store %72, %9[%69, %68] : memref<?x4xf32>
    }
    %52 = arith.addi %arg7, %c42 : index
    %53 = arith.remsi %52, %arg13 : index
    %54 = affine.apply affine_map<(d0) -> (d0 * 512)>(%53)
    %55 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%53)[%dim_1]
    %56 = arith.cmpi slt, %arg10, %55 : index
    scf.if %56 {
      %67 = arith.addi %arg10, %54 : index
      memref.store %c0_i32, %14[%67] : memref<?xi32>
    }
    %57 = arith.addi %arg7, %c49 : index
    %58 = arith.remsi %57, %arg13 : index
    %59 = affine.apply affine_map<(d0) -> (d0 * 512)>(%58)
    %60 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%58)[%dim_1]
    %61 = arith.cmpi slt, %arg10, %60 : index
    scf.if %61 {
      %67 = arith.addi %arg10, %59 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg5[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %14[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %14[%67] : memref<?xi32>
      }
    }
    %62 = arith.addi %arg7, %c56 : index
    %63 = arith.remsi %62, %arg13 : index
    %64 = affine.apply affine_map<(d0) -> (d0 * 512)>(%63)
    %65 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%63)[%18]
    %66 = arith.cmpi slt, %arg10, %65 : index
    scf.if %66 {
      %67 = arith.addi %arg10, %64 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %14[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg6[%71, %68] : memref<?x4xf32>
      memref.store %72, %16[%69, %68] : memref<?x4xf32>
    }
    gpu.terminator
  } {SCFToGPU_visited}
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before AFGPUFrontendOpt (af-gpu-frontend-opt) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c7 = arith.constant 7 : index
  %c14 = arith.constant 14 : index
  %c21 = arith.constant 21 : index
  %c28 = arith.constant 28 : index
  %c35 = arith.constant 35 : index
  %c42 = arith.constant 42 : index
  %c49 = arith.constant 49 : index
  %c56 = arith.constant 56 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = affine.apply affine_map<()[s0] -> (s0 ceildiv 512)>()[%21]
  gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
    %23 = arith.remsi %arg7, %arg13 : index
    %24 = affine.apply affine_map<(d0) -> (d0 * 512)>(%23)
    %25 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%23)[%dim]
    %26 = arith.cmpi slt, %arg10, %25 : index
    scf.if %26 {
      %67 = arith.addi %arg10, %24 : index
      memref.store %c0_i32, %0[%67] : memref<?xi32>
    }
    %27 = arith.addi %arg7, %c7 : index
    %28 = arith.remsi %27, %arg13 : index
    %29 = affine.apply affine_map<(d0) -> (d0 * 512)>(%28)
    %30 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%28)[%dim]
    %31 = arith.cmpi slt, %arg10, %30 : index
    scf.if %31 {
      %67 = arith.addi %arg10, %29 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg1[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %0[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %0[%67] : memref<?xi32>
      }
    }
    %32 = arith.addi %arg7, %c14 : index
    %33 = arith.remsi %32, %arg13 : index
    %34 = affine.apply affine_map<(d0) -> (d0 * 512)>(%33)
    %35 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%33)[%4]
    %36 = arith.cmpi slt, %arg10, %35 : index
    scf.if %36 {
      %67 = arith.addi %arg10, %34 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %0[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg2[%71, %68] : memref<?x4xf32>
      memref.store %72, %2[%69, %68] : memref<?x4xf32>
    }
    %37 = arith.addi %arg7, %c21 : index
    %38 = arith.remsi %37, %arg13 : index
    %39 = affine.apply affine_map<(d0) -> (d0 * 512)>(%38)
    %40 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%38)[%dim_0]
    %41 = arith.cmpi slt, %arg10, %40 : index
    scf.if %41 {
      %67 = arith.addi %arg10, %39 : index
      memref.store %c0_i32, %7[%67] : memref<?xi32>
    }
    %42 = arith.addi %arg7, %c28 : index
    %43 = arith.remsi %42, %arg13 : index
    %44 = affine.apply affine_map<(d0) -> (d0 * 512)>(%43)
    %45 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%43)[%dim_0]
    %46 = arith.cmpi slt, %arg10, %45 : index
    scf.if %46 {
      %67 = arith.addi %arg10, %44 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg3[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %7[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %7[%67] : memref<?xi32>
      }
    }
    %47 = arith.addi %arg7, %c35 : index
    %48 = arith.remsi %47, %arg13 : index
    %49 = affine.apply affine_map<(d0) -> (d0 * 512)>(%48)
    %50 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%48)[%11]
    %51 = arith.cmpi slt, %arg10, %50 : index
    scf.if %51 {
      %67 = arith.addi %arg10, %49 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %7[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg4[%71, %68] : memref<?x4xf32>
      memref.store %72, %9[%69, %68] : memref<?x4xf32>
    }
    %52 = arith.addi %arg7, %c42 : index
    %53 = arith.remsi %52, %arg13 : index
    %54 = affine.apply affine_map<(d0) -> (d0 * 512)>(%53)
    %55 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%53)[%dim_1]
    %56 = arith.cmpi slt, %arg10, %55 : index
    scf.if %56 {
      %67 = arith.addi %arg10, %54 : index
      memref.store %c0_i32, %14[%67] : memref<?xi32>
    }
    %57 = arith.addi %arg7, %c49 : index
    %58 = arith.remsi %57, %arg13 : index
    %59 = affine.apply affine_map<(d0) -> (d0 * 512)>(%58)
    %60 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%58)[%dim_1]
    %61 = arith.cmpi slt, %arg10, %60 : index
    scf.if %61 {
      %67 = arith.addi %arg10, %59 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg5[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %14[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %14[%67] : memref<?xi32>
      }
    }
    %62 = arith.addi %arg7, %c56 : index
    %63 = arith.remsi %62, %arg13 : index
    %64 = affine.apply affine_map<(d0) -> (d0 * 512)>(%63)
    %65 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%63)[%18]
    %66 = arith.cmpi slt, %arg10, %65 : index
    scf.if %66 {
      %67 = arith.addi %arg10, %64 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %14[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg6[%71, %68] : memref<?x4xf32>
      memref.store %72, %16[%69, %68] : memref<?x4xf32>
    }
    gpu.terminator
  } {SCFToGPU_visited}
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After AFGPUFrontendOpt (af-gpu-frontend-opt) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c7 = arith.constant 7 : index
  %c14 = arith.constant 14 : index
  %c21 = arith.constant 21 : index
  %c28 = arith.constant 28 : index
  %c35 = arith.constant 35 : index
  %c42 = arith.constant 42 : index
  %c49 = arith.constant 49 : index
  %c56 = arith.constant 56 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = affine.apply affine_map<()[s0] -> (s0 ceildiv 512)>()[%21]
  gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
    %23 = arith.remsi %arg7, %arg13 : index
    %24 = affine.apply affine_map<(d0) -> (d0 * 512)>(%23)
    %25 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%23)[%dim]
    %26 = arith.cmpi slt, %arg10, %25 : index
    scf.if %26 {
      %67 = arith.addi %arg10, %24 : index
      memref.store %c0_i32, %0[%67] : memref<?xi32>
    }
    %27 = arith.addi %arg7, %c7 : index
    %28 = arith.remsi %27, %arg13 : index
    %29 = affine.apply affine_map<(d0) -> (d0 * 512)>(%28)
    %30 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%28)[%dim]
    %31 = arith.cmpi slt, %arg10, %30 : index
    scf.if %31 {
      %67 = arith.addi %arg10, %29 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg1[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %0[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %0[%67] : memref<?xi32>
      }
    }
    %32 = arith.addi %arg7, %c14 : index
    %33 = arith.remsi %32, %arg13 : index
    %34 = affine.apply affine_map<(d0) -> (d0 * 512)>(%33)
    %35 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%33)[%4]
    %36 = arith.cmpi slt, %arg10, %35 : index
    scf.if %36 {
      %67 = arith.addi %arg10, %34 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %0[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg2[%71, %68] : memref<?x4xf32>
      memref.store %72, %2[%69, %68] : memref<?x4xf32>
    }
    %37 = arith.addi %arg7, %c21 : index
    %38 = arith.remsi %37, %arg13 : index
    %39 = affine.apply affine_map<(d0) -> (d0 * 512)>(%38)
    %40 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%38)[%dim_0]
    %41 = arith.cmpi slt, %arg10, %40 : index
    scf.if %41 {
      %67 = arith.addi %arg10, %39 : index
      memref.store %c0_i32, %7[%67] : memref<?xi32>
    }
    %42 = arith.addi %arg7, %c28 : index
    %43 = arith.remsi %42, %arg13 : index
    %44 = affine.apply affine_map<(d0) -> (d0 * 512)>(%43)
    %45 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%43)[%dim_0]
    %46 = arith.cmpi slt, %arg10, %45 : index
    scf.if %46 {
      %67 = arith.addi %arg10, %44 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg3[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %7[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %7[%67] : memref<?xi32>
      }
    }
    %47 = arith.addi %arg7, %c35 : index
    %48 = arith.remsi %47, %arg13 : index
    %49 = affine.apply affine_map<(d0) -> (d0 * 512)>(%48)
    %50 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%48)[%11]
    %51 = arith.cmpi slt, %arg10, %50 : index
    scf.if %51 {
      %67 = arith.addi %arg10, %49 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %7[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg4[%71, %68] : memref<?x4xf32>
      memref.store %72, %9[%69, %68] : memref<?x4xf32>
    }
    %52 = arith.addi %arg7, %c42 : index
    %53 = arith.remsi %52, %arg13 : index
    %54 = affine.apply affine_map<(d0) -> (d0 * 512)>(%53)
    %55 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%53)[%dim_1]
    %56 = arith.cmpi slt, %arg10, %55 : index
    scf.if %56 {
      %67 = arith.addi %arg10, %54 : index
      memref.store %c0_i32, %14[%67] : memref<?xi32>
    }
    %57 = arith.addi %arg7, %c49 : index
    %58 = arith.remsi %57, %arg13 : index
    %59 = affine.apply affine_map<(d0) -> (d0 * 512)>(%58)
    %60 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%58)[%dim_1]
    %61 = arith.cmpi slt, %arg10, %60 : index
    scf.if %61 {
      %67 = arith.addi %arg10, %59 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg5[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %14[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %14[%67] : memref<?xi32>
      }
    }
    %62 = arith.addi %arg7, %c56 : index
    %63 = arith.remsi %62, %arg13 : index
    %64 = affine.apply affine_map<(d0) -> (d0 * 512)>(%63)
    %65 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%63)[%18]
    %66 = arith.cmpi slt, %arg10, %65 : index
    scf.if %66 {
      %67 = arith.addi %arg10, %64 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %14[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg6[%71, %68] : memref<?x4xf32>
      memref.store %72, %16[%69, %68] : memref<?x4xf32>
    }
    gpu.terminator
  } {SCFToGPU_visited}
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before SCFForLoopSpecialization (scf-for-loop-specialization) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c7 = arith.constant 7 : index
  %c14 = arith.constant 14 : index
  %c21 = arith.constant 21 : index
  %c28 = arith.constant 28 : index
  %c35 = arith.constant 35 : index
  %c42 = arith.constant 42 : index
  %c49 = arith.constant 49 : index
  %c56 = arith.constant 56 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = affine.apply affine_map<()[s0] -> (s0 ceildiv 512)>()[%21]
  gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
    %23 = arith.remsi %arg7, %arg13 : index
    %24 = affine.apply affine_map<(d0) -> (d0 * 512)>(%23)
    %25 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%23)[%dim]
    %26 = arith.cmpi slt, %arg10, %25 : index
    scf.if %26 {
      %67 = arith.addi %arg10, %24 : index
      memref.store %c0_i32, %0[%67] : memref<?xi32>
    }
    %27 = arith.addi %arg7, %c7 : index
    %28 = arith.remsi %27, %arg13 : index
    %29 = affine.apply affine_map<(d0) -> (d0 * 512)>(%28)
    %30 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%28)[%dim]
    %31 = arith.cmpi slt, %arg10, %30 : index
    scf.if %31 {
      %67 = arith.addi %arg10, %29 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg1[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %0[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %0[%67] : memref<?xi32>
      }
    }
    %32 = arith.addi %arg7, %c14 : index
    %33 = arith.remsi %32, %arg13 : index
    %34 = affine.apply affine_map<(d0) -> (d0 * 512)>(%33)
    %35 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%33)[%4]
    %36 = arith.cmpi slt, %arg10, %35 : index
    scf.if %36 {
      %67 = arith.addi %arg10, %34 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %0[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg2[%71, %68] : memref<?x4xf32>
      memref.store %72, %2[%69, %68] : memref<?x4xf32>
    }
    %37 = arith.addi %arg7, %c21 : index
    %38 = arith.remsi %37, %arg13 : index
    %39 = affine.apply affine_map<(d0) -> (d0 * 512)>(%38)
    %40 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%38)[%dim_0]
    %41 = arith.cmpi slt, %arg10, %40 : index
    scf.if %41 {
      %67 = arith.addi %arg10, %39 : index
      memref.store %c0_i32, %7[%67] : memref<?xi32>
    }
    %42 = arith.addi %arg7, %c28 : index
    %43 = arith.remsi %42, %arg13 : index
    %44 = affine.apply affine_map<(d0) -> (d0 * 512)>(%43)
    %45 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%43)[%dim_0]
    %46 = arith.cmpi slt, %arg10, %45 : index
    scf.if %46 {
      %67 = arith.addi %arg10, %44 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg3[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %7[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %7[%67] : memref<?xi32>
      }
    }
    %47 = arith.addi %arg7, %c35 : index
    %48 = arith.remsi %47, %arg13 : index
    %49 = affine.apply affine_map<(d0) -> (d0 * 512)>(%48)
    %50 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%48)[%11]
    %51 = arith.cmpi slt, %arg10, %50 : index
    scf.if %51 {
      %67 = arith.addi %arg10, %49 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %7[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg4[%71, %68] : memref<?x4xf32>
      memref.store %72, %9[%69, %68] : memref<?x4xf32>
    }
    %52 = arith.addi %arg7, %c42 : index
    %53 = arith.remsi %52, %arg13 : index
    %54 = affine.apply affine_map<(d0) -> (d0 * 512)>(%53)
    %55 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%53)[%dim_1]
    %56 = arith.cmpi slt, %arg10, %55 : index
    scf.if %56 {
      %67 = arith.addi %arg10, %54 : index
      memref.store %c0_i32, %14[%67] : memref<?xi32>
    }
    %57 = arith.addi %arg7, %c49 : index
    %58 = arith.remsi %57, %arg13 : index
    %59 = affine.apply affine_map<(d0) -> (d0 * 512)>(%58)
    %60 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%58)[%dim_1]
    %61 = arith.cmpi slt, %arg10, %60 : index
    scf.if %61 {
      %67 = arith.addi %arg10, %59 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg5[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %14[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %14[%67] : memref<?xi32>
      }
    }
    %62 = arith.addi %arg7, %c56 : index
    %63 = arith.remsi %62, %arg13 : index
    %64 = affine.apply affine_map<(d0) -> (d0 * 512)>(%63)
    %65 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%63)[%18]
    %66 = arith.cmpi slt, %arg10, %65 : index
    scf.if %66 {
      %67 = arith.addi %arg10, %64 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %14[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg6[%71, %68] : memref<?x4xf32>
      memref.store %72, %16[%69, %68] : memref<?x4xf32>
    }
    gpu.terminator
  } {SCFToGPU_visited}
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After SCFForLoopSpecialization (scf-for-loop-specialization) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c7 = arith.constant 7 : index
  %c14 = arith.constant 14 : index
  %c21 = arith.constant 21 : index
  %c28 = arith.constant 28 : index
  %c35 = arith.constant 35 : index
  %c42 = arith.constant 42 : index
  %c49 = arith.constant 49 : index
  %c56 = arith.constant 56 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = affine.apply affine_map<()[s0] -> (s0 ceildiv 512)>()[%21]
  gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
    %23 = arith.remsi %arg7, %arg13 : index
    %24 = affine.apply affine_map<(d0) -> (d0 * 512)>(%23)
    %25 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%23)[%dim]
    %26 = arith.cmpi slt, %arg10, %25 : index
    scf.if %26 {
      %67 = arith.addi %arg10, %24 : index
      memref.store %c0_i32, %0[%67] : memref<?xi32>
    }
    %27 = arith.addi %arg7, %c7 : index
    %28 = arith.remsi %27, %arg13 : index
    %29 = affine.apply affine_map<(d0) -> (d0 * 512)>(%28)
    %30 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%28)[%dim]
    %31 = arith.cmpi slt, %arg10, %30 : index
    scf.if %31 {
      %67 = arith.addi %arg10, %29 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg1[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %0[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %0[%67] : memref<?xi32>
      }
    }
    %32 = arith.addi %arg7, %c14 : index
    %33 = arith.remsi %32, %arg13 : index
    %34 = affine.apply affine_map<(d0) -> (d0 * 512)>(%33)
    %35 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%33)[%4]
    %36 = arith.cmpi slt, %arg10, %35 : index
    scf.if %36 {
      %67 = arith.addi %arg10, %34 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %0[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg2[%71, %68] : memref<?x4xf32>
      memref.store %72, %2[%69, %68] : memref<?x4xf32>
    }
    %37 = arith.addi %arg7, %c21 : index
    %38 = arith.remsi %37, %arg13 : index
    %39 = affine.apply affine_map<(d0) -> (d0 * 512)>(%38)
    %40 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%38)[%dim_0]
    %41 = arith.cmpi slt, %arg10, %40 : index
    scf.if %41 {
      %67 = arith.addi %arg10, %39 : index
      memref.store %c0_i32, %7[%67] : memref<?xi32>
    }
    %42 = arith.addi %arg7, %c28 : index
    %43 = arith.remsi %42, %arg13 : index
    %44 = affine.apply affine_map<(d0) -> (d0 * 512)>(%43)
    %45 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%43)[%dim_0]
    %46 = arith.cmpi slt, %arg10, %45 : index
    scf.if %46 {
      %67 = arith.addi %arg10, %44 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg3[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %7[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %7[%67] : memref<?xi32>
      }
    }
    %47 = arith.addi %arg7, %c35 : index
    %48 = arith.remsi %47, %arg13 : index
    %49 = affine.apply affine_map<(d0) -> (d0 * 512)>(%48)
    %50 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%48)[%11]
    %51 = arith.cmpi slt, %arg10, %50 : index
    scf.if %51 {
      %67 = arith.addi %arg10, %49 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %7[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg4[%71, %68] : memref<?x4xf32>
      memref.store %72, %9[%69, %68] : memref<?x4xf32>
    }
    %52 = arith.addi %arg7, %c42 : index
    %53 = arith.remsi %52, %arg13 : index
    %54 = affine.apply affine_map<(d0) -> (d0 * 512)>(%53)
    %55 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%53)[%dim_1]
    %56 = arith.cmpi slt, %arg10, %55 : index
    scf.if %56 {
      %67 = arith.addi %arg10, %54 : index
      memref.store %c0_i32, %14[%67] : memref<?xi32>
    }
    %57 = arith.addi %arg7, %c49 : index
    %58 = arith.remsi %57, %arg13 : index
    %59 = affine.apply affine_map<(d0) -> (d0 * 512)>(%58)
    %60 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%58)[%dim_1]
    %61 = arith.cmpi slt, %arg10, %60 : index
    scf.if %61 {
      %67 = arith.addi %arg10, %59 : index
      scf.for %arg19 = %c0 to %c2 step %c1 {
        %68 = memref.load %arg5[%67, %arg19] : memref<?x2xi32>
        %69 = memref.load %14[%67] : memref<?xi32>
        %70 = arith.addi %69, %68 : i32
        memref.store %70, %14[%67] : memref<?xi32>
      }
    }
    %62 = arith.addi %arg7, %c56 : index
    %63 = arith.remsi %62, %arg13 : index
    %64 = affine.apply affine_map<(d0) -> (d0 * 512)>(%63)
    %65 = affine.min affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>(%63)[%18]
    %66 = arith.cmpi slt, %arg10, %65 : index
    scf.if %66 {
      %67 = arith.addi %arg10, %64 : index
      %68 = arith.remsi %67, %c4 : index
      %69 = arith.divsi %67, %c4 : index
      %70 = memref.load %14[%69] : memref<?xi32>
      %71 = arith.index_cast %70 : i32 to index
      %72 = memref.load %arg6[%71, %68] : memref<?x4xf32>
      memref.store %72, %16[%69, %68] : memref<?x4xf32>
    }
    gpu.terminator
  } {SCFToGPU_visited}
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before GpuLaunchSinkIndexComputations (gpu-launch-sink-index-computations) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 512)>
#map1 = affine_map<(d0) -> (d0 * 512)>
#map2 = affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %c14 = arith.constant 14 : index
    %c21 = arith.constant 21 : index
    %c28 = arith.constant 28 : index
    %c35 = arith.constant 35 : index
    %c42 = arith.constant 42 : index
    %c49 = arith.constant 49 : index
    %c56 = arith.constant 56 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = affine.apply #map()[%21]
    gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
      %23 = arith.remsi %arg7, %arg13 : index
      %24 = affine.apply #map1(%23)
      %25 = affine.min #map2(%23)[%dim]
      %26 = arith.cmpi slt, %arg10, %25 : index
      scf.if %26 {
        %67 = arith.addi %arg10, %24 : index
        memref.store %c0_i32, %0[%67] : memref<?xi32>
      }
      %27 = arith.addi %arg7, %c7 : index
      %28 = arith.remsi %27, %arg13 : index
      %29 = affine.apply #map1(%28)
      %30 = affine.min #map2(%28)[%dim]
      %31 = arith.cmpi slt, %arg10, %30 : index
      scf.if %31 {
        %67 = arith.addi %arg10, %29 : index
        scf.for %arg19 = %c0 to %c2 step %c1 {
          %68 = memref.load %arg1[%67, %arg19] : memref<?x2xi32>
          %69 = memref.load %0[%67] : memref<?xi32>
          %70 = arith.addi %69, %68 : i32
          memref.store %70, %0[%67] : memref<?xi32>
        }
      }
      %32 = arith.addi %arg7, %c14 : index
      %33 = arith.remsi %32, %arg13 : index
      %34 = affine.apply #map1(%33)
      %35 = affine.min #map2(%33)[%4]
      %36 = arith.cmpi slt, %arg10, %35 : index
      scf.if %36 {
        %67 = arith.addi %arg10, %34 : index
        %68 = arith.remsi %67, %c4 : index
        %69 = arith.divsi %67, %c4 : index
        %70 = memref.load %0[%69] : memref<?xi32>
        %71 = arith.index_cast %70 : i32 to index
        %72 = memref.load %arg2[%71, %68] : memref<?x4xf32>
        memref.store %72, %2[%69, %68] : memref<?x4xf32>
      }
      %37 = arith.addi %arg7, %c21 : index
      %38 = arith.remsi %37, %arg13 : index
      %39 = affine.apply #map1(%38)
      %40 = affine.min #map2(%38)[%dim_0]
      %41 = arith.cmpi slt, %arg10, %40 : index
      scf.if %41 {
        %67 = arith.addi %arg10, %39 : index
        memref.store %c0_i32, %7[%67] : memref<?xi32>
      }
      %42 = arith.addi %arg7, %c28 : index
      %43 = arith.remsi %42, %arg13 : index
      %44 = affine.apply #map1(%43)
      %45 = affine.min #map2(%43)[%dim_0]
      %46 = arith.cmpi slt, %arg10, %45 : index
      scf.if %46 {
        %67 = arith.addi %arg10, %44 : index
        scf.for %arg19 = %c0 to %c2 step %c1 {
          %68 = memref.load %arg3[%67, %arg19] : memref<?x2xi32>
          %69 = memref.load %7[%67] : memref<?xi32>
          %70 = arith.addi %69, %68 : i32
          memref.store %70, %7[%67] : memref<?xi32>
        }
      }
      %47 = arith.addi %arg7, %c35 : index
      %48 = arith.remsi %47, %arg13 : index
      %49 = affine.apply #map1(%48)
      %50 = affine.min #map2(%48)[%11]
      %51 = arith.cmpi slt, %arg10, %50 : index
      scf.if %51 {
        %67 = arith.addi %arg10, %49 : index
        %68 = arith.remsi %67, %c4 : index
        %69 = arith.divsi %67, %c4 : index
        %70 = memref.load %7[%69] : memref<?xi32>
        %71 = arith.index_cast %70 : i32 to index
        %72 = memref.load %arg4[%71, %68] : memref<?x4xf32>
        memref.store %72, %9[%69, %68] : memref<?x4xf32>
      }
      %52 = arith.addi %arg7, %c42 : index
      %53 = arith.remsi %52, %arg13 : index
      %54 = affine.apply #map1(%53)
      %55 = affine.min #map2(%53)[%dim_1]
      %56 = arith.cmpi slt, %arg10, %55 : index
      scf.if %56 {
        %67 = arith.addi %arg10, %54 : index
        memref.store %c0_i32, %14[%67] : memref<?xi32>
      }
      %57 = arith.addi %arg7, %c49 : index
      %58 = arith.remsi %57, %arg13 : index
      %59 = affine.apply #map1(%58)
      %60 = affine.min #map2(%58)[%dim_1]
      %61 = arith.cmpi slt, %arg10, %60 : index
      scf.if %61 {
        %67 = arith.addi %arg10, %59 : index
        scf.for %arg19 = %c0 to %c2 step %c1 {
          %68 = memref.load %arg5[%67, %arg19] : memref<?x2xi32>
          %69 = memref.load %14[%67] : memref<?xi32>
          %70 = arith.addi %69, %68 : i32
          memref.store %70, %14[%67] : memref<?xi32>
        }
      }
      %62 = arith.addi %arg7, %c56 : index
      %63 = arith.remsi %62, %arg13 : index
      %64 = affine.apply #map1(%63)
      %65 = affine.min #map2(%63)[%18]
      %66 = arith.cmpi slt, %arg10, %65 : index
      scf.if %66 {
        %67 = arith.addi %arg10, %64 : index
        %68 = arith.remsi %67, %c4 : index
        %69 = arith.divsi %67, %c4 : index
        %70 = memref.load %14[%69] : memref<?xi32>
        %71 = arith.index_cast %70 : i32 to index
        %72 = memref.load %arg6[%71, %68] : memref<?x4xf32>
        memref.store %72, %16[%69, %68] : memref<?x4xf32>
      }
      gpu.terminator
    } {SCFToGPU_visited}
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump After GpuLaunchSinkIndexComputations (gpu-launch-sink-index-computations) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 512)>
#map1 = affine_map<(d0) -> (d0 * 512)>
#map2 = affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %c14 = arith.constant 14 : index
    %c21 = arith.constant 21 : index
    %c28 = arith.constant 28 : index
    %c35 = arith.constant 35 : index
    %c42 = arith.constant 42 : index
    %c49 = arith.constant 49 : index
    %c56 = arith.constant 56 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = affine.apply #map()[%21]
    gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
      %c0_2 = arith.constant 0 : index
      %dim_3 = memref.dim %arg1, %c0_2 : memref<?x2xi32>
      %c0_i32_4 = arith.constant 0 : i32
      %c7_5 = arith.constant 7 : index
      %c2_6 = arith.constant 2 : index
      %c1_7 = arith.constant 1 : index
      %c14_8 = arith.constant 14 : index
      %c4_9 = arith.constant 4 : index
      %c21_10 = arith.constant 21 : index
      %dim_11 = memref.dim %arg3, %c0_2 : memref<?x2xi32>
      %c28_12 = arith.constant 28 : index
      %c35_13 = arith.constant 35 : index
      %c42_14 = arith.constant 42 : index
      %dim_15 = memref.dim %arg5, %c0_2 : memref<?x2xi32>
      %c49_16 = arith.constant 49 : index
      %c56_17 = arith.constant 56 : index
      %23 = arith.remsi %arg7, %arg13 : index
      %24 = affine.apply #map1(%23)
      %25 = affine.min #map2(%23)[%dim_3]
      %26 = arith.cmpi slt, %arg10, %25 : index
      scf.if %26 {
        %67 = arith.addi %arg10, %24 : index
        memref.store %c0_i32_4, %0[%67] : memref<?xi32>
      }
      %27 = arith.addi %arg7, %c7_5 : index
      %28 = arith.remsi %27, %arg13 : index
      %29 = affine.apply #map1(%28)
      %30 = affine.min #map2(%28)[%dim_3]
      %31 = arith.cmpi slt, %arg10, %30 : index
      scf.if %31 {
        %67 = arith.addi %arg10, %29 : index
        scf.for %arg19 = %c0_2 to %c2_6 step %c1_7 {
          %68 = memref.load %arg1[%67, %arg19] : memref<?x2xi32>
          %69 = memref.load %0[%67] : memref<?xi32>
          %70 = arith.addi %69, %68 : i32
          memref.store %70, %0[%67] : memref<?xi32>
        }
      }
      %32 = arith.addi %arg7, %c14_8 : index
      %33 = arith.remsi %32, %arg13 : index
      %34 = affine.apply #map1(%33)
      %35 = affine.min #map2(%33)[%4]
      %36 = arith.cmpi slt, %arg10, %35 : index
      scf.if %36 {
        %67 = arith.addi %arg10, %34 : index
        %68 = arith.remsi %67, %c4_9 : index
        %69 = arith.divsi %67, %c4_9 : index
        %70 = memref.load %0[%69] : memref<?xi32>
        %71 = arith.index_cast %70 : i32 to index
        %72 = memref.load %arg2[%71, %68] : memref<?x4xf32>
        memref.store %72, %2[%69, %68] : memref<?x4xf32>
      }
      %37 = arith.addi %arg7, %c21_10 : index
      %38 = arith.remsi %37, %arg13 : index
      %39 = affine.apply #map1(%38)
      %40 = affine.min #map2(%38)[%dim_11]
      %41 = arith.cmpi slt, %arg10, %40 : index
      scf.if %41 {
        %67 = arith.addi %arg10, %39 : index
        memref.store %c0_i32_4, %7[%67] : memref<?xi32>
      }
      %42 = arith.addi %arg7, %c28_12 : index
      %43 = arith.remsi %42, %arg13 : index
      %44 = affine.apply #map1(%43)
      %45 = affine.min #map2(%43)[%dim_11]
      %46 = arith.cmpi slt, %arg10, %45 : index
      scf.if %46 {
        %67 = arith.addi %arg10, %44 : index
        scf.for %arg19 = %c0_2 to %c2_6 step %c1_7 {
          %68 = memref.load %arg3[%67, %arg19] : memref<?x2xi32>
          %69 = memref.load %7[%67] : memref<?xi32>
          %70 = arith.addi %69, %68 : i32
          memref.store %70, %7[%67] : memref<?xi32>
        }
      }
      %47 = arith.addi %arg7, %c35_13 : index
      %48 = arith.remsi %47, %arg13 : index
      %49 = affine.apply #map1(%48)
      %50 = affine.min #map2(%48)[%11]
      %51 = arith.cmpi slt, %arg10, %50 : index
      scf.if %51 {
        %67 = arith.addi %arg10, %49 : index
        %68 = arith.remsi %67, %c4_9 : index
        %69 = arith.divsi %67, %c4_9 : index
        %70 = memref.load %7[%69] : memref<?xi32>
        %71 = arith.index_cast %70 : i32 to index
        %72 = memref.load %arg4[%71, %68] : memref<?x4xf32>
        memref.store %72, %9[%69, %68] : memref<?x4xf32>
      }
      %52 = arith.addi %arg7, %c42_14 : index
      %53 = arith.remsi %52, %arg13 : index
      %54 = affine.apply #map1(%53)
      %55 = affine.min #map2(%53)[%dim_15]
      %56 = arith.cmpi slt, %arg10, %55 : index
      scf.if %56 {
        %67 = arith.addi %arg10, %54 : index
        memref.store %c0_i32_4, %14[%67] : memref<?xi32>
      }
      %57 = arith.addi %arg7, %c49_16 : index
      %58 = arith.remsi %57, %arg13 : index
      %59 = affine.apply #map1(%58)
      %60 = affine.min #map2(%58)[%dim_15]
      %61 = arith.cmpi slt, %arg10, %60 : index
      scf.if %61 {
        %67 = arith.addi %arg10, %59 : index
        scf.for %arg19 = %c0_2 to %c2_6 step %c1_7 {
          %68 = memref.load %arg5[%67, %arg19] : memref<?x2xi32>
          %69 = memref.load %14[%67] : memref<?xi32>
          %70 = arith.addi %69, %68 : i32
          memref.store %70, %14[%67] : memref<?xi32>
        }
      }
      %62 = arith.addi %arg7, %c56_17 : index
      %63 = arith.remsi %62, %arg13 : index
      %64 = affine.apply #map1(%63)
      %65 = affine.min #map2(%63)[%18]
      %66 = arith.cmpi slt, %arg10, %65 : index
      scf.if %66 {
        %67 = arith.addi %arg10, %64 : index
        %68 = arith.remsi %67, %c4_9 : index
        %69 = arith.divsi %67, %c4_9 : index
        %70 = memref.load %14[%69] : memref<?xi32>
        %71 = arith.index_cast %70 : i32 to index
        %72 = memref.load %arg6[%71, %68] : memref<?x4xf32>
        memref.store %72, %16[%69, %68] : memref<?x4xf32>
      }
      gpu.terminator
    } {SCFToGPU_visited}
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump Before GpuKernelOutlining (gpu-kernel-outlining) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 512)>
#map1 = affine_map<(d0) -> (d0 * 512)>
#map2 = affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>
module {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %c14 = arith.constant 14 : index
    %c21 = arith.constant 21 : index
    %c28 = arith.constant 28 : index
    %c35 = arith.constant 35 : index
    %c42 = arith.constant 42 : index
    %c49 = arith.constant 49 : index
    %c56 = arith.constant 56 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = affine.apply #map()[%21]
    gpu.launch blocks(%arg7, %arg8, %arg9) in (%arg13 = %22, %arg14 = %c1, %arg15 = %c1) threads(%arg10, %arg11, %arg12) in (%arg16 = %c512, %arg17 = %c1, %arg18 = %c1) {
      %c0_2 = arith.constant 0 : index
      %dim_3 = memref.dim %arg1, %c0_2 : memref<?x2xi32>
      %c0_i32_4 = arith.constant 0 : i32
      %c7_5 = arith.constant 7 : index
      %c2_6 = arith.constant 2 : index
      %c1_7 = arith.constant 1 : index
      %c14_8 = arith.constant 14 : index
      %c4_9 = arith.constant 4 : index
      %c21_10 = arith.constant 21 : index
      %dim_11 = memref.dim %arg3, %c0_2 : memref<?x2xi32>
      %c28_12 = arith.constant 28 : index
      %c35_13 = arith.constant 35 : index
      %c42_14 = arith.constant 42 : index
      %dim_15 = memref.dim %arg5, %c0_2 : memref<?x2xi32>
      %c49_16 = arith.constant 49 : index
      %c56_17 = arith.constant 56 : index
      %23 = arith.remsi %arg7, %arg13 : index
      %24 = affine.apply #map1(%23)
      %25 = affine.min #map2(%23)[%dim_3]
      %26 = arith.cmpi slt, %arg10, %25 : index
      scf.if %26 {
        %67 = arith.addi %arg10, %24 : index
        memref.store %c0_i32_4, %0[%67] : memref<?xi32>
      }
      %27 = arith.addi %arg7, %c7_5 : index
      %28 = arith.remsi %27, %arg13 : index
      %29 = affine.apply #map1(%28)
      %30 = affine.min #map2(%28)[%dim_3]
      %31 = arith.cmpi slt, %arg10, %30 : index
      scf.if %31 {
        %67 = arith.addi %arg10, %29 : index
        scf.for %arg19 = %c0_2 to %c2_6 step %c1_7 {
          %68 = memref.load %arg1[%67, %arg19] : memref<?x2xi32>
          %69 = memref.load %0[%67] : memref<?xi32>
          %70 = arith.addi %69, %68 : i32
          memref.store %70, %0[%67] : memref<?xi32>
        }
      }
      %32 = arith.addi %arg7, %c14_8 : index
      %33 = arith.remsi %32, %arg13 : index
      %34 = affine.apply #map1(%33)
      %35 = affine.min #map2(%33)[%4]
      %36 = arith.cmpi slt, %arg10, %35 : index
      scf.if %36 {
        %67 = arith.addi %arg10, %34 : index
        %68 = arith.remsi %67, %c4_9 : index
        %69 = arith.divsi %67, %c4_9 : index
        %70 = memref.load %0[%69] : memref<?xi32>
        %71 = arith.index_cast %70 : i32 to index
        %72 = memref.load %arg2[%71, %68] : memref<?x4xf32>
        memref.store %72, %2[%69, %68] : memref<?x4xf32>
      }
      %37 = arith.addi %arg7, %c21_10 : index
      %38 = arith.remsi %37, %arg13 : index
      %39 = affine.apply #map1(%38)
      %40 = affine.min #map2(%38)[%dim_11]
      %41 = arith.cmpi slt, %arg10, %40 : index
      scf.if %41 {
        %67 = arith.addi %arg10, %39 : index
        memref.store %c0_i32_4, %7[%67] : memref<?xi32>
      }
      %42 = arith.addi %arg7, %c28_12 : index
      %43 = arith.remsi %42, %arg13 : index
      %44 = affine.apply #map1(%43)
      %45 = affine.min #map2(%43)[%dim_11]
      %46 = arith.cmpi slt, %arg10, %45 : index
      scf.if %46 {
        %67 = arith.addi %arg10, %44 : index
        scf.for %arg19 = %c0_2 to %c2_6 step %c1_7 {
          %68 = memref.load %arg3[%67, %arg19] : memref<?x2xi32>
          %69 = memref.load %7[%67] : memref<?xi32>
          %70 = arith.addi %69, %68 : i32
          memref.store %70, %7[%67] : memref<?xi32>
        }
      }
      %47 = arith.addi %arg7, %c35_13 : index
      %48 = arith.remsi %47, %arg13 : index
      %49 = affine.apply #map1(%48)
      %50 = affine.min #map2(%48)[%11]
      %51 = arith.cmpi slt, %arg10, %50 : index
      scf.if %51 {
        %67 = arith.addi %arg10, %49 : index
        %68 = arith.remsi %67, %c4_9 : index
        %69 = arith.divsi %67, %c4_9 : index
        %70 = memref.load %7[%69] : memref<?xi32>
        %71 = arith.index_cast %70 : i32 to index
        %72 = memref.load %arg4[%71, %68] : memref<?x4xf32>
        memref.store %72, %9[%69, %68] : memref<?x4xf32>
      }
      %52 = arith.addi %arg7, %c42_14 : index
      %53 = arith.remsi %52, %arg13 : index
      %54 = affine.apply #map1(%53)
      %55 = affine.min #map2(%53)[%dim_15]
      %56 = arith.cmpi slt, %arg10, %55 : index
      scf.if %56 {
        %67 = arith.addi %arg10, %54 : index
        memref.store %c0_i32_4, %14[%67] : memref<?xi32>
      }
      %57 = arith.addi %arg7, %c49_16 : index
      %58 = arith.remsi %57, %arg13 : index
      %59 = affine.apply #map1(%58)
      %60 = affine.min #map2(%58)[%dim_15]
      %61 = arith.cmpi slt, %arg10, %60 : index
      scf.if %61 {
        %67 = arith.addi %arg10, %59 : index
        scf.for %arg19 = %c0_2 to %c2_6 step %c1_7 {
          %68 = memref.load %arg5[%67, %arg19] : memref<?x2xi32>
          %69 = memref.load %14[%67] : memref<?xi32>
          %70 = arith.addi %69, %68 : i32
          memref.store %70, %14[%67] : memref<?xi32>
        }
      }
      %62 = arith.addi %arg7, %c56_17 : index
      %63 = arith.remsi %62, %arg13 : index
      %64 = affine.apply #map1(%63)
      %65 = affine.min #map2(%63)[%18]
      %66 = arith.cmpi slt, %arg10, %65 : index
      scf.if %66 {
        %67 = arith.addi %arg10, %64 : index
        %68 = arith.remsi %67, %c4_9 : index
        %69 = arith.divsi %67, %c4_9 : index
        %70 = memref.load %14[%69] : memref<?xi32>
        %71 = arith.index_cast %70 : i32 to index
        %72 = memref.load %arg6[%71, %68] : memref<?x4xf32>
        memref.store %72, %16[%69, %68] : memref<?x4xf32>
      }
      gpu.terminator
    } {SCFToGPU_visited}
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump After GpuKernelOutlining (gpu-kernel-outlining) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 512)>
#map1 = affine_map<(d0) -> (d0 * 512)>
#map2 = affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>
module attributes {gpu.container_module} {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %c14 = arith.constant 14 : index
    %c21 = arith.constant 21 : index
    %c28 = arith.constant 28 : index
    %c35 = arith.constant 35 : index
    %c42 = arith.constant 42 : index
    %c49 = arith.constant 49 : index
    %c56 = arith.constant 56 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = affine.apply #map()[%21]
    gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%22, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
  gpu.module @predict_online_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_0_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?x2xi32>, %arg2: memref<?x2xi32>, %arg3: memref<?xi32>, %arg4: index, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>, %arg7: memref<?xi32>, %arg8: index, %arg9: memref<?x4xf32>, %arg10: memref<?x4xf32>, %arg11: memref<?xi32>, %arg12: index, %arg13: memref<?x4xf32>, %arg14: memref<?x4xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
      %c0_i32 = arith.constant 0 : i32
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %c4 = arith.constant 4 : index
      %c21 = arith.constant 21 : index
      %dim_0 = memref.dim %arg1, %c0 : memref<?x2xi32>
      %c28 = arith.constant 28 : index
      %c35 = arith.constant 35 : index
      %c42 = arith.constant 42 : index
      %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
      %c49 = arith.constant 49 : index
      %c56 = arith.constant 56 : index
      %12 = arith.remsi %0, %6 : index
      %13 = affine.apply #map1(%12)
      %14 = affine.min #map2(%12)[%dim]
      %15 = arith.cmpi slt, %3, %14 : index
      scf.if %15 {
        %56 = arith.addi %3, %13 : index
        memref.store %c0_i32, %arg3[%56] : memref<?xi32>
      }
      %16 = arith.addi %0, %c7 : index
      %17 = arith.remsi %16, %6 : index
      %18 = affine.apply #map1(%17)
      %19 = affine.min #map2(%17)[%dim]
      %20 = arith.cmpi slt, %3, %19 : index
      scf.if %20 {
        %56 = arith.addi %3, %18 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %57 = memref.load %arg0[%56, %arg15] : memref<?x2xi32>
          %58 = memref.load %arg3[%56] : memref<?xi32>
          %59 = arith.addi %58, %57 : i32
          memref.store %59, %arg3[%56] : memref<?xi32>
        }
      }
      %21 = arith.addi %0, %c14 : index
      %22 = arith.remsi %21, %6 : index
      %23 = affine.apply #map1(%22)
      %24 = affine.min #map2(%22)[%arg4]
      %25 = arith.cmpi slt, %3, %24 : index
      scf.if %25 {
        %56 = arith.addi %3, %23 : index
        %57 = arith.remsi %56, %c4 : index
        %58 = arith.divsi %56, %c4 : index
        %59 = memref.load %arg3[%58] : memref<?xi32>
        %60 = arith.index_cast %59 : i32 to index
        %61 = memref.load %arg5[%60, %57] : memref<?x4xf32>
        memref.store %61, %arg6[%58, %57] : memref<?x4xf32>
      }
      %26 = arith.addi %0, %c21 : index
      %27 = arith.remsi %26, %6 : index
      %28 = affine.apply #map1(%27)
      %29 = affine.min #map2(%27)[%dim_0]
      %30 = arith.cmpi slt, %3, %29 : index
      scf.if %30 {
        %56 = arith.addi %3, %28 : index
        memref.store %c0_i32, %arg7[%56] : memref<?xi32>
      }
      %31 = arith.addi %0, %c28 : index
      %32 = arith.remsi %31, %6 : index
      %33 = affine.apply #map1(%32)
      %34 = affine.min #map2(%32)[%dim_0]
      %35 = arith.cmpi slt, %3, %34 : index
      scf.if %35 {
        %56 = arith.addi %3, %33 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %57 = memref.load %arg1[%56, %arg15] : memref<?x2xi32>
          %58 = memref.load %arg7[%56] : memref<?xi32>
          %59 = arith.addi %58, %57 : i32
          memref.store %59, %arg7[%56] : memref<?xi32>
        }
      }
      %36 = arith.addi %0, %c35 : index
      %37 = arith.remsi %36, %6 : index
      %38 = affine.apply #map1(%37)
      %39 = affine.min #map2(%37)[%arg8]
      %40 = arith.cmpi slt, %3, %39 : index
      scf.if %40 {
        %56 = arith.addi %3, %38 : index
        %57 = arith.remsi %56, %c4 : index
        %58 = arith.divsi %56, %c4 : index
        %59 = memref.load %arg7[%58] : memref<?xi32>
        %60 = arith.index_cast %59 : i32 to index
        %61 = memref.load %arg9[%60, %57] : memref<?x4xf32>
        memref.store %61, %arg10[%58, %57] : memref<?x4xf32>
      }
      %41 = arith.addi %0, %c42 : index
      %42 = arith.remsi %41, %6 : index
      %43 = affine.apply #map1(%42)
      %44 = affine.min #map2(%42)[%dim_1]
      %45 = arith.cmpi slt, %3, %44 : index
      scf.if %45 {
        %56 = arith.addi %3, %43 : index
        memref.store %c0_i32, %arg11[%56] : memref<?xi32>
      }
      %46 = arith.addi %0, %c49 : index
      %47 = arith.remsi %46, %6 : index
      %48 = affine.apply #map1(%47)
      %49 = affine.min #map2(%47)[%dim_1]
      %50 = arith.cmpi slt, %3, %49 : index
      scf.if %50 {
        %56 = arith.addi %3, %48 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %57 = memref.load %arg2[%56, %arg15] : memref<?x2xi32>
          %58 = memref.load %arg11[%56] : memref<?xi32>
          %59 = arith.addi %58, %57 : i32
          memref.store %59, %arg11[%56] : memref<?xi32>
        }
      }
      %51 = arith.addi %0, %c56 : index
      %52 = arith.remsi %51, %6 : index
      %53 = affine.apply #map1(%52)
      %54 = affine.min #map2(%52)[%arg12]
      %55 = arith.cmpi slt, %3, %54 : index
      scf.if %55 {
        %56 = arith.addi %3, %53 : index
        %57 = arith.remsi %56, %c4 : index
        %58 = arith.divsi %56, %c4 : index
        %59 = memref.load %arg11[%58] : memref<?xi32>
        %60 = arith.index_cast %59 : i32 to index
        %61 = memref.load %arg13[%60, %57] : memref<?x4xf32>
        memref.store %61, %arg14[%58, %57] : memref<?x4xf32>
      }
      gpu.return
    }
  }
}


// -----// IR Dump Before AFGPUAllocaToGPUMem (af-gpu-alloca-to-gpu-mem) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 512)>
#map1 = affine_map<(d0) -> (d0 * 512)>
#map2 = affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>
module attributes {gpu.container_module} {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c7 = arith.constant 7 : index
    %c14 = arith.constant 14 : index
    %c21 = arith.constant 21 : index
    %c28 = arith.constant 28 : index
    %c35 = arith.constant 35 : index
    %c42 = arith.constant 42 : index
    %c49 = arith.constant 49 : index
    %c56 = arith.constant 56 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = affine.apply #map()[%21]
    gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%22, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
  gpu.module @predict_online_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_0_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?x2xi32>, %arg2: memref<?x2xi32>, %arg3: memref<?xi32>, %arg4: index, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>, %arg7: memref<?xi32>, %arg8: index, %arg9: memref<?x4xf32>, %arg10: memref<?x4xf32>, %arg11: memref<?xi32>, %arg12: index, %arg13: memref<?x4xf32>, %arg14: memref<?x4xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c0 = arith.constant 0 : index
      %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
      %c0_i32 = arith.constant 0 : i32
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %c4 = arith.constant 4 : index
      %c21 = arith.constant 21 : index
      %dim_0 = memref.dim %arg1, %c0 : memref<?x2xi32>
      %c28 = arith.constant 28 : index
      %c35 = arith.constant 35 : index
      %c42 = arith.constant 42 : index
      %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
      %c49 = arith.constant 49 : index
      %c56 = arith.constant 56 : index
      %12 = arith.remsi %0, %6 : index
      %13 = affine.apply #map1(%12)
      %14 = affine.min #map2(%12)[%dim]
      %15 = arith.cmpi slt, %3, %14 : index
      scf.if %15 {
        %56 = arith.addi %3, %13 : index
        memref.store %c0_i32, %arg3[%56] : memref<?xi32>
      }
      %16 = arith.addi %0, %c7 : index
      %17 = arith.remsi %16, %6 : index
      %18 = affine.apply #map1(%17)
      %19 = affine.min #map2(%17)[%dim]
      %20 = arith.cmpi slt, %3, %19 : index
      scf.if %20 {
        %56 = arith.addi %3, %18 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %57 = memref.load %arg0[%56, %arg15] : memref<?x2xi32>
          %58 = memref.load %arg3[%56] : memref<?xi32>
          %59 = arith.addi %58, %57 : i32
          memref.store %59, %arg3[%56] : memref<?xi32>
        }
      }
      %21 = arith.addi %0, %c14 : index
      %22 = arith.remsi %21, %6 : index
      %23 = affine.apply #map1(%22)
      %24 = affine.min #map2(%22)[%arg4]
      %25 = arith.cmpi slt, %3, %24 : index
      scf.if %25 {
        %56 = arith.addi %3, %23 : index
        %57 = arith.remsi %56, %c4 : index
        %58 = arith.divsi %56, %c4 : index
        %59 = memref.load %arg3[%58] : memref<?xi32>
        %60 = arith.index_cast %59 : i32 to index
        %61 = memref.load %arg5[%60, %57] : memref<?x4xf32>
        memref.store %61, %arg6[%58, %57] : memref<?x4xf32>
      }
      %26 = arith.addi %0, %c21 : index
      %27 = arith.remsi %26, %6 : index
      %28 = affine.apply #map1(%27)
      %29 = affine.min #map2(%27)[%dim_0]
      %30 = arith.cmpi slt, %3, %29 : index
      scf.if %30 {
        %56 = arith.addi %3, %28 : index
        memref.store %c0_i32, %arg7[%56] : memref<?xi32>
      }
      %31 = arith.addi %0, %c28 : index
      %32 = arith.remsi %31, %6 : index
      %33 = affine.apply #map1(%32)
      %34 = affine.min #map2(%32)[%dim_0]
      %35 = arith.cmpi slt, %3, %34 : index
      scf.if %35 {
        %56 = arith.addi %3, %33 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %57 = memref.load %arg1[%56, %arg15] : memref<?x2xi32>
          %58 = memref.load %arg7[%56] : memref<?xi32>
          %59 = arith.addi %58, %57 : i32
          memref.store %59, %arg7[%56] : memref<?xi32>
        }
      }
      %36 = arith.addi %0, %c35 : index
      %37 = arith.remsi %36, %6 : index
      %38 = affine.apply #map1(%37)
      %39 = affine.min #map2(%37)[%arg8]
      %40 = arith.cmpi slt, %3, %39 : index
      scf.if %40 {
        %56 = arith.addi %3, %38 : index
        %57 = arith.remsi %56, %c4 : index
        %58 = arith.divsi %56, %c4 : index
        %59 = memref.load %arg7[%58] : memref<?xi32>
        %60 = arith.index_cast %59 : i32 to index
        %61 = memref.load %arg9[%60, %57] : memref<?x4xf32>
        memref.store %61, %arg10[%58, %57] : memref<?x4xf32>
      }
      %41 = arith.addi %0, %c42 : index
      %42 = arith.remsi %41, %6 : index
      %43 = affine.apply #map1(%42)
      %44 = affine.min #map2(%42)[%dim_1]
      %45 = arith.cmpi slt, %3, %44 : index
      scf.if %45 {
        %56 = arith.addi %3, %43 : index
        memref.store %c0_i32, %arg11[%56] : memref<?xi32>
      }
      %46 = arith.addi %0, %c49 : index
      %47 = arith.remsi %46, %6 : index
      %48 = affine.apply #map1(%47)
      %49 = affine.min #map2(%47)[%dim_1]
      %50 = arith.cmpi slt, %3, %49 : index
      scf.if %50 {
        %56 = arith.addi %3, %48 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %57 = memref.load %arg2[%56, %arg15] : memref<?x2xi32>
          %58 = memref.load %arg11[%56] : memref<?xi32>
          %59 = arith.addi %58, %57 : i32
          memref.store %59, %arg11[%56] : memref<?xi32>
        }
      }
      %51 = arith.addi %0, %c56 : index
      %52 = arith.remsi %51, %6 : index
      %53 = affine.apply #map1(%52)
      %54 = affine.min #map2(%52)[%arg12]
      %55 = arith.cmpi slt, %3, %54 : index
      scf.if %55 {
        %56 = arith.addi %3, %53 : index
        %57 = arith.remsi %56, %c4 : index
        %58 = arith.divsi %56, %c4 : index
        %59 = memref.load %arg11[%58] : memref<?xi32>
        %60 = arith.index_cast %59 : i32 to index
        %61 = memref.load %arg13[%60, %57] : memref<?x4xf32>
        memref.store %61, %arg14[%58, %57] : memref<?x4xf32>
      }
      gpu.return
    }
  }
}


// -----// IR Dump After AFGPUAllocaToGPUMem (af-gpu-alloca-to-gpu-mem) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 512)>
#map1 = affine_map<(d0) -> (d0 * 512)>
#map2 = affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>
module attributes {gpu.container_module} {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = affine.apply #map()[%21]
    gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%22, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
  gpu.module @predict_online_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_0_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?x2xi32>, %arg2: memref<?x2xi32>, %arg3: memref<?xi32>, %arg4: index, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>, %arg7: memref<?xi32>, %arg8: index, %arg9: memref<?x4xf32>, %arg10: memref<?x4xf32>, %arg11: memref<?xi32>, %arg12: index, %arg13: memref<?x4xf32>, %arg14: memref<?x4xf32>) kernel {
      %c56 = arith.constant 56 : index
      %c49 = arith.constant 49 : index
      %c42 = arith.constant 42 : index
      %c35 = arith.constant 35 : index
      %c28 = arith.constant 28 : index
      %c21 = arith.constant 21 : index
      %c4 = arith.constant 4 : index
      %c14 = arith.constant 14 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.grid_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
      %dim_0 = memref.dim %arg1, %c0 : memref<?x2xi32>
      %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
      %3 = arith.remsi %0, %2 : index
      %4 = affine.apply #map1(%3)
      %5 = affine.min #map2(%3)[%dim]
      %6 = arith.cmpi slt, %1, %5 : index
      scf.if %6 {
        %47 = arith.addi %1, %4 : index
        memref.store %c0_i32, %arg3[%47] : memref<?xi32>
      }
      %7 = arith.addi %0, %c7 : index
      %8 = arith.remsi %7, %2 : index
      %9 = affine.apply #map1(%8)
      %10 = affine.min #map2(%8)[%dim]
      %11 = arith.cmpi slt, %1, %10 : index
      scf.if %11 {
        %47 = arith.addi %1, %9 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %48 = memref.load %arg0[%47, %arg15] : memref<?x2xi32>
          %49 = memref.load %arg3[%47] : memref<?xi32>
          %50 = arith.addi %49, %48 : i32
          memref.store %50, %arg3[%47] : memref<?xi32>
        }
      }
      %12 = arith.addi %0, %c14 : index
      %13 = arith.remsi %12, %2 : index
      %14 = affine.apply #map1(%13)
      %15 = affine.min #map2(%13)[%arg4]
      %16 = arith.cmpi slt, %1, %15 : index
      scf.if %16 {
        %47 = arith.addi %1, %14 : index
        %48 = arith.remsi %47, %c4 : index
        %49 = arith.divsi %47, %c4 : index
        %50 = memref.load %arg3[%49] : memref<?xi32>
        %51 = arith.index_cast %50 : i32 to index
        %52 = memref.load %arg5[%51, %48] : memref<?x4xf32>
        memref.store %52, %arg6[%49, %48] : memref<?x4xf32>
      }
      %17 = arith.addi %0, %c21 : index
      %18 = arith.remsi %17, %2 : index
      %19 = affine.apply #map1(%18)
      %20 = affine.min #map2(%18)[%dim_0]
      %21 = arith.cmpi slt, %1, %20 : index
      scf.if %21 {
        %47 = arith.addi %1, %19 : index
        memref.store %c0_i32, %arg7[%47] : memref<?xi32>
      }
      %22 = arith.addi %0, %c28 : index
      %23 = arith.remsi %22, %2 : index
      %24 = affine.apply #map1(%23)
      %25 = affine.min #map2(%23)[%dim_0]
      %26 = arith.cmpi slt, %1, %25 : index
      scf.if %26 {
        %47 = arith.addi %1, %24 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %48 = memref.load %arg1[%47, %arg15] : memref<?x2xi32>
          %49 = memref.load %arg7[%47] : memref<?xi32>
          %50 = arith.addi %49, %48 : i32
          memref.store %50, %arg7[%47] : memref<?xi32>
        }
      }
      %27 = arith.addi %0, %c35 : index
      %28 = arith.remsi %27, %2 : index
      %29 = affine.apply #map1(%28)
      %30 = affine.min #map2(%28)[%arg8]
      %31 = arith.cmpi slt, %1, %30 : index
      scf.if %31 {
        %47 = arith.addi %1, %29 : index
        %48 = arith.remsi %47, %c4 : index
        %49 = arith.divsi %47, %c4 : index
        %50 = memref.load %arg7[%49] : memref<?xi32>
        %51 = arith.index_cast %50 : i32 to index
        %52 = memref.load %arg9[%51, %48] : memref<?x4xf32>
        memref.store %52, %arg10[%49, %48] : memref<?x4xf32>
      }
      %32 = arith.addi %0, %c42 : index
      %33 = arith.remsi %32, %2 : index
      %34 = affine.apply #map1(%33)
      %35 = affine.min #map2(%33)[%dim_1]
      %36 = arith.cmpi slt, %1, %35 : index
      scf.if %36 {
        %47 = arith.addi %1, %34 : index
        memref.store %c0_i32, %arg11[%47] : memref<?xi32>
      }
      %37 = arith.addi %0, %c49 : index
      %38 = arith.remsi %37, %2 : index
      %39 = affine.apply #map1(%38)
      %40 = affine.min #map2(%38)[%dim_1]
      %41 = arith.cmpi slt, %1, %40 : index
      scf.if %41 {
        %47 = arith.addi %1, %39 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %48 = memref.load %arg2[%47, %arg15] : memref<?x2xi32>
          %49 = memref.load %arg11[%47] : memref<?xi32>
          %50 = arith.addi %49, %48 : i32
          memref.store %50, %arg11[%47] : memref<?xi32>
        }
      }
      %42 = arith.addi %0, %c56 : index
      %43 = arith.remsi %42, %2 : index
      %44 = affine.apply #map1(%43)
      %45 = affine.min #map2(%43)[%arg12]
      %46 = arith.cmpi slt, %1, %45 : index
      scf.if %46 {
        %47 = arith.addi %1, %44 : index
        %48 = arith.remsi %47, %c4 : index
        %49 = arith.divsi %47, %c4 : index
        %50 = memref.load %arg11[%49] : memref<?xi32>
        %51 = arith.index_cast %50 : i32 to index
        %52 = memref.load %arg13[%51, %48] : memref<?x4xf32>
        memref.store %52, %arg14[%49, %48] : memref<?x4xf32>
      }
      gpu.return
    }
  }
}


// -----// IR Dump Before ConvertAffineToStandard (lower-affine) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 512)>
#map1 = affine_map<(d0) -> (d0 * 512)>
#map2 = affine_map<(d0)[s0] -> (d0 * -512 + s0, 512)>
module attributes {gpu.container_module} {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = affine.apply #map()[%21]
    gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%22, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
  gpu.module @predict_online_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_0_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?x2xi32>, %arg2: memref<?x2xi32>, %arg3: memref<?xi32>, %arg4: index, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>, %arg7: memref<?xi32>, %arg8: index, %arg9: memref<?x4xf32>, %arg10: memref<?x4xf32>, %arg11: memref<?xi32>, %arg12: index, %arg13: memref<?x4xf32>, %arg14: memref<?x4xf32>) kernel {
      %c56 = arith.constant 56 : index
      %c49 = arith.constant 49 : index
      %c42 = arith.constant 42 : index
      %c35 = arith.constant 35 : index
      %c28 = arith.constant 28 : index
      %c21 = arith.constant 21 : index
      %c4 = arith.constant 4 : index
      %c14 = arith.constant 14 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.grid_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
      %dim_0 = memref.dim %arg1, %c0 : memref<?x2xi32>
      %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
      %3 = arith.remsi %0, %2 : index
      %4 = affine.apply #map1(%3)
      %5 = affine.min #map2(%3)[%dim]
      %6 = arith.cmpi slt, %1, %5 : index
      scf.if %6 {
        %47 = arith.addi %1, %4 : index
        memref.store %c0_i32, %arg3[%47] : memref<?xi32>
      }
      %7 = arith.addi %0, %c7 : index
      %8 = arith.remsi %7, %2 : index
      %9 = affine.apply #map1(%8)
      %10 = affine.min #map2(%8)[%dim]
      %11 = arith.cmpi slt, %1, %10 : index
      scf.if %11 {
        %47 = arith.addi %1, %9 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %48 = memref.load %arg0[%47, %arg15] : memref<?x2xi32>
          %49 = memref.load %arg3[%47] : memref<?xi32>
          %50 = arith.addi %49, %48 : i32
          memref.store %50, %arg3[%47] : memref<?xi32>
        }
      }
      %12 = arith.addi %0, %c14 : index
      %13 = arith.remsi %12, %2 : index
      %14 = affine.apply #map1(%13)
      %15 = affine.min #map2(%13)[%arg4]
      %16 = arith.cmpi slt, %1, %15 : index
      scf.if %16 {
        %47 = arith.addi %1, %14 : index
        %48 = arith.remsi %47, %c4 : index
        %49 = arith.divsi %47, %c4 : index
        %50 = memref.load %arg3[%49] : memref<?xi32>
        %51 = arith.index_cast %50 : i32 to index
        %52 = memref.load %arg5[%51, %48] : memref<?x4xf32>
        memref.store %52, %arg6[%49, %48] : memref<?x4xf32>
      }
      %17 = arith.addi %0, %c21 : index
      %18 = arith.remsi %17, %2 : index
      %19 = affine.apply #map1(%18)
      %20 = affine.min #map2(%18)[%dim_0]
      %21 = arith.cmpi slt, %1, %20 : index
      scf.if %21 {
        %47 = arith.addi %1, %19 : index
        memref.store %c0_i32, %arg7[%47] : memref<?xi32>
      }
      %22 = arith.addi %0, %c28 : index
      %23 = arith.remsi %22, %2 : index
      %24 = affine.apply #map1(%23)
      %25 = affine.min #map2(%23)[%dim_0]
      %26 = arith.cmpi slt, %1, %25 : index
      scf.if %26 {
        %47 = arith.addi %1, %24 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %48 = memref.load %arg1[%47, %arg15] : memref<?x2xi32>
          %49 = memref.load %arg7[%47] : memref<?xi32>
          %50 = arith.addi %49, %48 : i32
          memref.store %50, %arg7[%47] : memref<?xi32>
        }
      }
      %27 = arith.addi %0, %c35 : index
      %28 = arith.remsi %27, %2 : index
      %29 = affine.apply #map1(%28)
      %30 = affine.min #map2(%28)[%arg8]
      %31 = arith.cmpi slt, %1, %30 : index
      scf.if %31 {
        %47 = arith.addi %1, %29 : index
        %48 = arith.remsi %47, %c4 : index
        %49 = arith.divsi %47, %c4 : index
        %50 = memref.load %arg7[%49] : memref<?xi32>
        %51 = arith.index_cast %50 : i32 to index
        %52 = memref.load %arg9[%51, %48] : memref<?x4xf32>
        memref.store %52, %arg10[%49, %48] : memref<?x4xf32>
      }
      %32 = arith.addi %0, %c42 : index
      %33 = arith.remsi %32, %2 : index
      %34 = affine.apply #map1(%33)
      %35 = affine.min #map2(%33)[%dim_1]
      %36 = arith.cmpi slt, %1, %35 : index
      scf.if %36 {
        %47 = arith.addi %1, %34 : index
        memref.store %c0_i32, %arg11[%47] : memref<?xi32>
      }
      %37 = arith.addi %0, %c49 : index
      %38 = arith.remsi %37, %2 : index
      %39 = affine.apply #map1(%38)
      %40 = affine.min #map2(%38)[%dim_1]
      %41 = arith.cmpi slt, %1, %40 : index
      scf.if %41 {
        %47 = arith.addi %1, %39 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %48 = memref.load %arg2[%47, %arg15] : memref<?x2xi32>
          %49 = memref.load %arg11[%47] : memref<?xi32>
          %50 = arith.addi %49, %48 : i32
          memref.store %50, %arg11[%47] : memref<?xi32>
        }
      }
      %42 = arith.addi %0, %c56 : index
      %43 = arith.remsi %42, %2 : index
      %44 = affine.apply #map1(%43)
      %45 = affine.min #map2(%43)[%arg12]
      %46 = arith.cmpi slt, %1, %45 : index
      scf.if %46 {
        %47 = arith.addi %1, %44 : index
        %48 = arith.remsi %47, %c4 : index
        %49 = arith.divsi %47, %c4 : index
        %50 = memref.load %arg11[%49] : memref<?xi32>
        %51 = arith.index_cast %50 : i32 to index
        %52 = memref.load %arg13[%51, %48] : memref<?x4xf32>
        memref.store %52, %arg14[%49, %48] : memref<?x4xf32>
      }
      gpu.return
    }
  }
}


// -----// IR Dump After ConvertAffineToStandard (lower-affine) //----- //
module attributes {gpu.container_module} {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %c512_2 = arith.constant 512 : index
    %c0_3 = arith.constant 0 : index
    %c1_4 = arith.constant 1 : index
    %22 = arith.cmpi sle, %21, %c0_3 : index
    %23 = arith.subi %c0_3, %21 : index
    %24 = arith.subi %21, %c1_4 : index
    %25 = arith.select %22, %23, %24 : index
    %26 = arith.divsi %25, %c512_2 : index
    %27 = arith.subi %c0_3, %26 : index
    %28 = arith.addi %26, %c1_4 : index
    %29 = arith.select %22, %27, %28 : index
    gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%29, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
  gpu.module @predict_online_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_0_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?x2xi32>, %arg2: memref<?x2xi32>, %arg3: memref<?xi32>, %arg4: index, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>, %arg7: memref<?xi32>, %arg8: index, %arg9: memref<?x4xf32>, %arg10: memref<?x4xf32>, %arg11: memref<?xi32>, %arg12: index, %arg13: memref<?x4xf32>, %arg14: memref<?x4xf32>) kernel {
      %c56 = arith.constant 56 : index
      %c49 = arith.constant 49 : index
      %c42 = arith.constant 42 : index
      %c35 = arith.constant 35 : index
      %c28 = arith.constant 28 : index
      %c21 = arith.constant 21 : index
      %c4 = arith.constant 4 : index
      %c14 = arith.constant 14 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.grid_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
      %dim_0 = memref.dim %arg1, %c0 : memref<?x2xi32>
      %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
      %3 = arith.remsi %0, %2 : index
      %c512 = arith.constant 512 : index
      %4 = arith.muli %3, %c512 : index
      %c-512 = arith.constant -512 : index
      %5 = arith.muli %3, %c-512 : index
      %6 = arith.addi %5, %dim : index
      %c512_2 = arith.constant 512 : index
      %7 = arith.cmpi slt, %6, %c512_2 : index
      %8 = arith.select %7, %6, %c512_2 : index
      %9 = arith.cmpi slt, %1, %8 : index
      scf.if %9 {
        %74 = arith.addi %1, %4 : index
        memref.store %c0_i32, %arg3[%74] : memref<?xi32>
      }
      %10 = arith.addi %0, %c7 : index
      %11 = arith.remsi %10, %2 : index
      %c512_3 = arith.constant 512 : index
      %12 = arith.muli %11, %c512_3 : index
      %c-512_4 = arith.constant -512 : index
      %13 = arith.muli %11, %c-512_4 : index
      %14 = arith.addi %13, %dim : index
      %c512_5 = arith.constant 512 : index
      %15 = arith.cmpi slt, %14, %c512_5 : index
      %16 = arith.select %15, %14, %c512_5 : index
      %17 = arith.cmpi slt, %1, %16 : index
      scf.if %17 {
        %74 = arith.addi %1, %12 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %75 = memref.load %arg0[%74, %arg15] : memref<?x2xi32>
          %76 = memref.load %arg3[%74] : memref<?xi32>
          %77 = arith.addi %76, %75 : i32
          memref.store %77, %arg3[%74] : memref<?xi32>
        }
      }
      %18 = arith.addi %0, %c14 : index
      %19 = arith.remsi %18, %2 : index
      %c512_6 = arith.constant 512 : index
      %20 = arith.muli %19, %c512_6 : index
      %c-512_7 = arith.constant -512 : index
      %21 = arith.muli %19, %c-512_7 : index
      %22 = arith.addi %21, %arg4 : index
      %c512_8 = arith.constant 512 : index
      %23 = arith.cmpi slt, %22, %c512_8 : index
      %24 = arith.select %23, %22, %c512_8 : index
      %25 = arith.cmpi slt, %1, %24 : index
      scf.if %25 {
        %74 = arith.addi %1, %20 : index
        %75 = arith.remsi %74, %c4 : index
        %76 = arith.divsi %74, %c4 : index
        %77 = memref.load %arg3[%76] : memref<?xi32>
        %78 = arith.index_cast %77 : i32 to index
        %79 = memref.load %arg5[%78, %75] : memref<?x4xf32>
        memref.store %79, %arg6[%76, %75] : memref<?x4xf32>
      }
      %26 = arith.addi %0, %c21 : index
      %27 = arith.remsi %26, %2 : index
      %c512_9 = arith.constant 512 : index
      %28 = arith.muli %27, %c512_9 : index
      %c-512_10 = arith.constant -512 : index
      %29 = arith.muli %27, %c-512_10 : index
      %30 = arith.addi %29, %dim_0 : index
      %c512_11 = arith.constant 512 : index
      %31 = arith.cmpi slt, %30, %c512_11 : index
      %32 = arith.select %31, %30, %c512_11 : index
      %33 = arith.cmpi slt, %1, %32 : index
      scf.if %33 {
        %74 = arith.addi %1, %28 : index
        memref.store %c0_i32, %arg7[%74] : memref<?xi32>
      }
      %34 = arith.addi %0, %c28 : index
      %35 = arith.remsi %34, %2 : index
      %c512_12 = arith.constant 512 : index
      %36 = arith.muli %35, %c512_12 : index
      %c-512_13 = arith.constant -512 : index
      %37 = arith.muli %35, %c-512_13 : index
      %38 = arith.addi %37, %dim_0 : index
      %c512_14 = arith.constant 512 : index
      %39 = arith.cmpi slt, %38, %c512_14 : index
      %40 = arith.select %39, %38, %c512_14 : index
      %41 = arith.cmpi slt, %1, %40 : index
      scf.if %41 {
        %74 = arith.addi %1, %36 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %75 = memref.load %arg1[%74, %arg15] : memref<?x2xi32>
          %76 = memref.load %arg7[%74] : memref<?xi32>
          %77 = arith.addi %76, %75 : i32
          memref.store %77, %arg7[%74] : memref<?xi32>
        }
      }
      %42 = arith.addi %0, %c35 : index
      %43 = arith.remsi %42, %2 : index
      %c512_15 = arith.constant 512 : index
      %44 = arith.muli %43, %c512_15 : index
      %c-512_16 = arith.constant -512 : index
      %45 = arith.muli %43, %c-512_16 : index
      %46 = arith.addi %45, %arg8 : index
      %c512_17 = arith.constant 512 : index
      %47 = arith.cmpi slt, %46, %c512_17 : index
      %48 = arith.select %47, %46, %c512_17 : index
      %49 = arith.cmpi slt, %1, %48 : index
      scf.if %49 {
        %74 = arith.addi %1, %44 : index
        %75 = arith.remsi %74, %c4 : index
        %76 = arith.divsi %74, %c4 : index
        %77 = memref.load %arg7[%76] : memref<?xi32>
        %78 = arith.index_cast %77 : i32 to index
        %79 = memref.load %arg9[%78, %75] : memref<?x4xf32>
        memref.store %79, %arg10[%76, %75] : memref<?x4xf32>
      }
      %50 = arith.addi %0, %c42 : index
      %51 = arith.remsi %50, %2 : index
      %c512_18 = arith.constant 512 : index
      %52 = arith.muli %51, %c512_18 : index
      %c-512_19 = arith.constant -512 : index
      %53 = arith.muli %51, %c-512_19 : index
      %54 = arith.addi %53, %dim_1 : index
      %c512_20 = arith.constant 512 : index
      %55 = arith.cmpi slt, %54, %c512_20 : index
      %56 = arith.select %55, %54, %c512_20 : index
      %57 = arith.cmpi slt, %1, %56 : index
      scf.if %57 {
        %74 = arith.addi %1, %52 : index
        memref.store %c0_i32, %arg11[%74] : memref<?xi32>
      }
      %58 = arith.addi %0, %c49 : index
      %59 = arith.remsi %58, %2 : index
      %c512_21 = arith.constant 512 : index
      %60 = arith.muli %59, %c512_21 : index
      %c-512_22 = arith.constant -512 : index
      %61 = arith.muli %59, %c-512_22 : index
      %62 = arith.addi %61, %dim_1 : index
      %c512_23 = arith.constant 512 : index
      %63 = arith.cmpi slt, %62, %c512_23 : index
      %64 = arith.select %63, %62, %c512_23 : index
      %65 = arith.cmpi slt, %1, %64 : index
      scf.if %65 {
        %74 = arith.addi %1, %60 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %75 = memref.load %arg2[%74, %arg15] : memref<?x2xi32>
          %76 = memref.load %arg11[%74] : memref<?xi32>
          %77 = arith.addi %76, %75 : i32
          memref.store %77, %arg11[%74] : memref<?xi32>
        }
      }
      %66 = arith.addi %0, %c56 : index
      %67 = arith.remsi %66, %2 : index
      %c512_24 = arith.constant 512 : index
      %68 = arith.muli %67, %c512_24 : index
      %c-512_25 = arith.constant -512 : index
      %69 = arith.muli %67, %c-512_25 : index
      %70 = arith.addi %69, %arg12 : index
      %c512_26 = arith.constant 512 : index
      %71 = arith.cmpi slt, %70, %c512_26 : index
      %72 = arith.select %71, %70, %c512_26 : index
      %73 = arith.cmpi slt, %1, %72 : index
      scf.if %73 {
        %74 = arith.addi %1, %68 : index
        %75 = arith.remsi %74, %c4 : index
        %76 = arith.divsi %74, %c4 : index
        %77 = memref.load %arg11[%76] : memref<?xi32>
        %78 = arith.index_cast %77 : i32 to index
        %79 = memref.load %arg13[%78, %75] : memref<?x4xf32>
        memref.store %79, %arg14[%76, %75] : memref<?x4xf32>
      }
      gpu.return
    }
  }
}


// -----// IR Dump Before ConvertShapeConstraints (convert-shape-constraints) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %c512_2 = arith.constant 512 : index
  %c0_3 = arith.constant 0 : index
  %c1_4 = arith.constant 1 : index
  %22 = arith.cmpi sle, %21, %c0_3 : index
  %23 = arith.subi %c0_3, %21 : index
  %24 = arith.subi %21, %c1_4 : index
  %25 = arith.select %22, %23, %24 : index
  %26 = arith.divsi %25, %c512_2 : index
  %27 = arith.subi %c0_3, %26 : index
  %28 = arith.addi %26, %c1_4 : index
  %29 = arith.select %22, %27, %28 : index
  gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%29, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After ConvertShapeConstraints (convert-shape-constraints) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = arith.cmpi sle, %21, %c0 : index
  %23 = arith.subi %c0, %21 : index
  %24 = arith.subi %21, %c1 : index
  %25 = arith.select %22, %23, %24 : index
  %26 = arith.divsi %25, %c512 : index
  %27 = arith.subi %c0, %26 : index
  %28 = arith.addi %26, %c1 : index
  %29 = arith.select %22, %27, %28 : index
  gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%29, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = arith.cmpi sle, %21, %c0 : index
  %23 = arith.subi %c0, %21 : index
  %24 = arith.subi %21, %c1 : index
  %25 = arith.select %22, %23, %24 : index
  %26 = arith.divsi %25, %c512 : index
  %27 = arith.subi %c0, %26 : index
  %28 = arith.addi %26, %c1 : index
  %29 = arith.select %22, %27, %28 : index
  gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%29, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c40960 = arith.constant 40960 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
  %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %4 = arith.muli %dim, %c4 : index
  %5 = arith.maxui %dim, %4 : index
  %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
  %6 = arith.maxui %5, %dim_0 : index
  %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
  %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
  %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %11 = arith.muli %dim_0, %c4 : index
  %12 = arith.maxui %6, %11 : index
  %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
  %13 = arith.maxui %12, %dim_1 : index
  %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
  %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
  tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
  %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
  tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %18 = arith.muli %dim_1, %c4 : index
  %19 = arith.maxui %13, %18 : index
  %20 = arith.cmpi ugt, %19, %c40960 : index
  %21 = arith.select %20, %19, %c40960 : index
  %22 = arith.cmpi sle, %21, %c0 : index
  %23 = arith.subi %c0, %21 : index
  %24 = arith.subi %21, %c1 : index
  %25 = arith.select %22, %23, %24 : index
  %26 = arith.divsi %25, %c512 : index
  %27 = arith.subi %c0, %26 : index
  %28 = arith.addi %26, %c1 : index
  %29 = arith.select %22, %27, %28 : index
  gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%29, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
  tf_framework.dealloc(%arg0, %14) : memref<?xi32>
  tf_framework.dealloc(%arg0, %7) : memref<?xi32>
  tf_framework.dealloc(%arg0, %0) : memref<?xi32>
  return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before SCFToControlFlow (convert-scf-to-cf) //----- //
module attributes {gpu.container_module} {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = arith.cmpi sle, %21, %c0 : index
    %23 = arith.subi %c0, %21 : index
    %24 = arith.subi %21, %c1 : index
    %25 = arith.select %22, %23, %24 : index
    %26 = arith.divsi %25, %c512 : index
    %27 = arith.subi %c0, %26 : index
    %28 = arith.addi %26, %c1 : index
    %29 = arith.select %22, %27, %28 : index
    gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%29, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
  gpu.module @predict_online_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_0_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?x2xi32>, %arg2: memref<?x2xi32>, %arg3: memref<?xi32>, %arg4: index, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>, %arg7: memref<?xi32>, %arg8: index, %arg9: memref<?x4xf32>, %arg10: memref<?x4xf32>, %arg11: memref<?xi32>, %arg12: index, %arg13: memref<?x4xf32>, %arg14: memref<?x4xf32>) kernel {
      %c56 = arith.constant 56 : index
      %c49 = arith.constant 49 : index
      %c42 = arith.constant 42 : index
      %c35 = arith.constant 35 : index
      %c28 = arith.constant 28 : index
      %c21 = arith.constant 21 : index
      %c4 = arith.constant 4 : index
      %c14 = arith.constant 14 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.grid_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
      %dim_0 = memref.dim %arg1, %c0 : memref<?x2xi32>
      %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
      %3 = arith.remsi %0, %2 : index
      %c512 = arith.constant 512 : index
      %4 = arith.muli %3, %c512 : index
      %c-512 = arith.constant -512 : index
      %5 = arith.muli %3, %c-512 : index
      %6 = arith.addi %5, %dim : index
      %c512_2 = arith.constant 512 : index
      %7 = arith.cmpi slt, %6, %c512_2 : index
      %8 = arith.select %7, %6, %c512_2 : index
      %9 = arith.cmpi slt, %1, %8 : index
      scf.if %9 {
        %74 = arith.addi %1, %4 : index
        memref.store %c0_i32, %arg3[%74] : memref<?xi32>
      }
      %10 = arith.addi %0, %c7 : index
      %11 = arith.remsi %10, %2 : index
      %c512_3 = arith.constant 512 : index
      %12 = arith.muli %11, %c512_3 : index
      %c-512_4 = arith.constant -512 : index
      %13 = arith.muli %11, %c-512_4 : index
      %14 = arith.addi %13, %dim : index
      %c512_5 = arith.constant 512 : index
      %15 = arith.cmpi slt, %14, %c512_5 : index
      %16 = arith.select %15, %14, %c512_5 : index
      %17 = arith.cmpi slt, %1, %16 : index
      scf.if %17 {
        %74 = arith.addi %1, %12 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %75 = memref.load %arg0[%74, %arg15] : memref<?x2xi32>
          %76 = memref.load %arg3[%74] : memref<?xi32>
          %77 = arith.addi %76, %75 : i32
          memref.store %77, %arg3[%74] : memref<?xi32>
        }
      }
      %18 = arith.addi %0, %c14 : index
      %19 = arith.remsi %18, %2 : index
      %c512_6 = arith.constant 512 : index
      %20 = arith.muli %19, %c512_6 : index
      %c-512_7 = arith.constant -512 : index
      %21 = arith.muli %19, %c-512_7 : index
      %22 = arith.addi %21, %arg4 : index
      %c512_8 = arith.constant 512 : index
      %23 = arith.cmpi slt, %22, %c512_8 : index
      %24 = arith.select %23, %22, %c512_8 : index
      %25 = arith.cmpi slt, %1, %24 : index
      scf.if %25 {
        %74 = arith.addi %1, %20 : index
        %75 = arith.remsi %74, %c4 : index
        %76 = arith.divsi %74, %c4 : index
        %77 = memref.load %arg3[%76] : memref<?xi32>
        %78 = arith.index_cast %77 : i32 to index
        %79 = memref.load %arg5[%78, %75] : memref<?x4xf32>
        memref.store %79, %arg6[%76, %75] : memref<?x4xf32>
      }
      %26 = arith.addi %0, %c21 : index
      %27 = arith.remsi %26, %2 : index
      %c512_9 = arith.constant 512 : index
      %28 = arith.muli %27, %c512_9 : index
      %c-512_10 = arith.constant -512 : index
      %29 = arith.muli %27, %c-512_10 : index
      %30 = arith.addi %29, %dim_0 : index
      %c512_11 = arith.constant 512 : index
      %31 = arith.cmpi slt, %30, %c512_11 : index
      %32 = arith.select %31, %30, %c512_11 : index
      %33 = arith.cmpi slt, %1, %32 : index
      scf.if %33 {
        %74 = arith.addi %1, %28 : index
        memref.store %c0_i32, %arg7[%74] : memref<?xi32>
      }
      %34 = arith.addi %0, %c28 : index
      %35 = arith.remsi %34, %2 : index
      %c512_12 = arith.constant 512 : index
      %36 = arith.muli %35, %c512_12 : index
      %c-512_13 = arith.constant -512 : index
      %37 = arith.muli %35, %c-512_13 : index
      %38 = arith.addi %37, %dim_0 : index
      %c512_14 = arith.constant 512 : index
      %39 = arith.cmpi slt, %38, %c512_14 : index
      %40 = arith.select %39, %38, %c512_14 : index
      %41 = arith.cmpi slt, %1, %40 : index
      scf.if %41 {
        %74 = arith.addi %1, %36 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %75 = memref.load %arg1[%74, %arg15] : memref<?x2xi32>
          %76 = memref.load %arg7[%74] : memref<?xi32>
          %77 = arith.addi %76, %75 : i32
          memref.store %77, %arg7[%74] : memref<?xi32>
        }
      }
      %42 = arith.addi %0, %c35 : index
      %43 = arith.remsi %42, %2 : index
      %c512_15 = arith.constant 512 : index
      %44 = arith.muli %43, %c512_15 : index
      %c-512_16 = arith.constant -512 : index
      %45 = arith.muli %43, %c-512_16 : index
      %46 = arith.addi %45, %arg8 : index
      %c512_17 = arith.constant 512 : index
      %47 = arith.cmpi slt, %46, %c512_17 : index
      %48 = arith.select %47, %46, %c512_17 : index
      %49 = arith.cmpi slt, %1, %48 : index
      scf.if %49 {
        %74 = arith.addi %1, %44 : index
        %75 = arith.remsi %74, %c4 : index
        %76 = arith.divsi %74, %c4 : index
        %77 = memref.load %arg7[%76] : memref<?xi32>
        %78 = arith.index_cast %77 : i32 to index
        %79 = memref.load %arg9[%78, %75] : memref<?x4xf32>
        memref.store %79, %arg10[%76, %75] : memref<?x4xf32>
      }
      %50 = arith.addi %0, %c42 : index
      %51 = arith.remsi %50, %2 : index
      %c512_18 = arith.constant 512 : index
      %52 = arith.muli %51, %c512_18 : index
      %c-512_19 = arith.constant -512 : index
      %53 = arith.muli %51, %c-512_19 : index
      %54 = arith.addi %53, %dim_1 : index
      %c512_20 = arith.constant 512 : index
      %55 = arith.cmpi slt, %54, %c512_20 : index
      %56 = arith.select %55, %54, %c512_20 : index
      %57 = arith.cmpi slt, %1, %56 : index
      scf.if %57 {
        %74 = arith.addi %1, %52 : index
        memref.store %c0_i32, %arg11[%74] : memref<?xi32>
      }
      %58 = arith.addi %0, %c49 : index
      %59 = arith.remsi %58, %2 : index
      %c512_21 = arith.constant 512 : index
      %60 = arith.muli %59, %c512_21 : index
      %c-512_22 = arith.constant -512 : index
      %61 = arith.muli %59, %c-512_22 : index
      %62 = arith.addi %61, %dim_1 : index
      %c512_23 = arith.constant 512 : index
      %63 = arith.cmpi slt, %62, %c512_23 : index
      %64 = arith.select %63, %62, %c512_23 : index
      %65 = arith.cmpi slt, %1, %64 : index
      scf.if %65 {
        %74 = arith.addi %1, %60 : index
        scf.for %arg15 = %c0 to %c2 step %c1 {
          %75 = memref.load %arg2[%74, %arg15] : memref<?x2xi32>
          %76 = memref.load %arg11[%74] : memref<?xi32>
          %77 = arith.addi %76, %75 : i32
          memref.store %77, %arg11[%74] : memref<?xi32>
        }
      }
      %66 = arith.addi %0, %c56 : index
      %67 = arith.remsi %66, %2 : index
      %c512_24 = arith.constant 512 : index
      %68 = arith.muli %67, %c512_24 : index
      %c-512_25 = arith.constant -512 : index
      %69 = arith.muli %67, %c-512_25 : index
      %70 = arith.addi %69, %arg12 : index
      %c512_26 = arith.constant 512 : index
      %71 = arith.cmpi slt, %70, %c512_26 : index
      %72 = arith.select %71, %70, %c512_26 : index
      %73 = arith.cmpi slt, %1, %72 : index
      scf.if %73 {
        %74 = arith.addi %1, %68 : index
        %75 = arith.remsi %74, %c4 : index
        %76 = arith.divsi %74, %c4 : index
        %77 = memref.load %arg11[%76] : memref<?xi32>
        %78 = arith.index_cast %77 : i32 to index
        %79 = memref.load %arg13[%78, %75] : memref<?x4xf32>
        memref.store %79, %arg14[%76, %75] : memref<?x4xf32>
      }
      gpu.return
    }
  }
}


// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
module attributes {gpu.container_module} {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = arith.cmpi sle, %21, %c0 : index
    %23 = arith.subi %c0, %21 : index
    %24 = arith.subi %21, %c1 : index
    %25 = arith.select %22, %23, %24 : index
    %26 = arith.divsi %25, %c512 : index
    %27 = arith.subi %c0, %26 : index
    %28 = arith.addi %26, %c1 : index
    %29 = arith.select %22, %27, %28 : index
    gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%29, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
  gpu.module @predict_online_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_0_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?x2xi32>, %arg2: memref<?x2xi32>, %arg3: memref<?xi32>, %arg4: index, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>, %arg7: memref<?xi32>, %arg8: index, %arg9: memref<?x4xf32>, %arg10: memref<?x4xf32>, %arg11: memref<?xi32>, %arg12: index, %arg13: memref<?x4xf32>, %arg14: memref<?x4xf32>) kernel {
      %c56 = arith.constant 56 : index
      %c49 = arith.constant 49 : index
      %c42 = arith.constant 42 : index
      %c35 = arith.constant 35 : index
      %c28 = arith.constant 28 : index
      %c21 = arith.constant 21 : index
      %c4 = arith.constant 4 : index
      %c14 = arith.constant 14 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.grid_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
      %dim_0 = memref.dim %arg1, %c0 : memref<?x2xi32>
      %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
      %3 = arith.remsi %0, %2 : index
      %c512 = arith.constant 512 : index
      %4 = arith.muli %3, %c512 : index
      %c-512 = arith.constant -512 : index
      %5 = arith.muli %3, %c-512 : index
      %6 = arith.addi %5, %dim : index
      %c512_2 = arith.constant 512 : index
      %7 = arith.cmpi slt, %6, %c512_2 : index
      %8 = arith.select %7, %6, %c512_2 : index
      %9 = arith.cmpi slt, %1, %8 : index
      cf.cond_br %9, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %10 = arith.addi %1, %4 : index
      memref.store %c0_i32, %arg3[%10] : memref<?xi32>
      cf.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      %11 = arith.addi %0, %c7 : index
      %12 = arith.remsi %11, %2 : index
      %c512_3 = arith.constant 512 : index
      %13 = arith.muli %12, %c512_3 : index
      %c-512_4 = arith.constant -512 : index
      %14 = arith.muli %12, %c-512_4 : index
      %15 = arith.addi %14, %dim : index
      %c512_5 = arith.constant 512 : index
      %16 = arith.cmpi slt, %15, %c512_5 : index
      %17 = arith.select %16, %15, %c512_5 : index
      %18 = arith.cmpi slt, %1, %17 : index
      cf.cond_br %18, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      %19 = arith.addi %1, %13 : index
      cf.br ^bb5(%c0 : index)
    ^bb5(%20: index):  // 2 preds: ^bb4, ^bb6
      %21 = arith.cmpi slt, %20, %c2 : index
      cf.cond_br %21, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %22 = memref.load %arg0[%19, %20] : memref<?x2xi32>
      %23 = memref.load %arg3[%19] : memref<?xi32>
      %24 = arith.addi %23, %22 : i32
      memref.store %24, %arg3[%19] : memref<?xi32>
      %25 = arith.addi %20, %c1 : index
      cf.br ^bb5(%25 : index)
    ^bb7:  // pred: ^bb5
      cf.br ^bb8
    ^bb8:  // 2 preds: ^bb3, ^bb7
      %26 = arith.addi %0, %c14 : index
      %27 = arith.remsi %26, %2 : index
      %c512_6 = arith.constant 512 : index
      %28 = arith.muli %27, %c512_6 : index
      %c-512_7 = arith.constant -512 : index
      %29 = arith.muli %27, %c-512_7 : index
      %30 = arith.addi %29, %arg4 : index
      %c512_8 = arith.constant 512 : index
      %31 = arith.cmpi slt, %30, %c512_8 : index
      %32 = arith.select %31, %30, %c512_8 : index
      %33 = arith.cmpi slt, %1, %32 : index
      cf.cond_br %33, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      %34 = arith.addi %1, %28 : index
      %35 = arith.remsi %34, %c4 : index
      %36 = arith.divsi %34, %c4 : index
      %37 = memref.load %arg3[%36] : memref<?xi32>
      %38 = arith.index_cast %37 : i32 to index
      %39 = memref.load %arg5[%38, %35] : memref<?x4xf32>
      memref.store %39, %arg6[%36, %35] : memref<?x4xf32>
      cf.br ^bb10
    ^bb10:  // 2 preds: ^bb8, ^bb9
      %40 = arith.addi %0, %c21 : index
      %41 = arith.remsi %40, %2 : index
      %c512_9 = arith.constant 512 : index
      %42 = arith.muli %41, %c512_9 : index
      %c-512_10 = arith.constant -512 : index
      %43 = arith.muli %41, %c-512_10 : index
      %44 = arith.addi %43, %dim_0 : index
      %c512_11 = arith.constant 512 : index
      %45 = arith.cmpi slt, %44, %c512_11 : index
      %46 = arith.select %45, %44, %c512_11 : index
      %47 = arith.cmpi slt, %1, %46 : index
      cf.cond_br %47, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      %48 = arith.addi %1, %42 : index
      memref.store %c0_i32, %arg7[%48] : memref<?xi32>
      cf.br ^bb12
    ^bb12:  // 2 preds: ^bb10, ^bb11
      %49 = arith.addi %0, %c28 : index
      %50 = arith.remsi %49, %2 : index
      %c512_12 = arith.constant 512 : index
      %51 = arith.muli %50, %c512_12 : index
      %c-512_13 = arith.constant -512 : index
      %52 = arith.muli %50, %c-512_13 : index
      %53 = arith.addi %52, %dim_0 : index
      %c512_14 = arith.constant 512 : index
      %54 = arith.cmpi slt, %53, %c512_14 : index
      %55 = arith.select %54, %53, %c512_14 : index
      %56 = arith.cmpi slt, %1, %55 : index
      cf.cond_br %56, ^bb13, ^bb17
    ^bb13:  // pred: ^bb12
      %57 = arith.addi %1, %51 : index
      cf.br ^bb14(%c0 : index)
    ^bb14(%58: index):  // 2 preds: ^bb13, ^bb15
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb15, ^bb16
    ^bb15:  // pred: ^bb14
      %60 = memref.load %arg1[%57, %58] : memref<?x2xi32>
      %61 = memref.load %arg7[%57] : memref<?xi32>
      %62 = arith.addi %61, %60 : i32
      memref.store %62, %arg7[%57] : memref<?xi32>
      %63 = arith.addi %58, %c1 : index
      cf.br ^bb14(%63 : index)
    ^bb16:  // pred: ^bb14
      cf.br ^bb17
    ^bb17:  // 2 preds: ^bb12, ^bb16
      %64 = arith.addi %0, %c35 : index
      %65 = arith.remsi %64, %2 : index
      %c512_15 = arith.constant 512 : index
      %66 = arith.muli %65, %c512_15 : index
      %c-512_16 = arith.constant -512 : index
      %67 = arith.muli %65, %c-512_16 : index
      %68 = arith.addi %67, %arg8 : index
      %c512_17 = arith.constant 512 : index
      %69 = arith.cmpi slt, %68, %c512_17 : index
      %70 = arith.select %69, %68, %c512_17 : index
      %71 = arith.cmpi slt, %1, %70 : index
      cf.cond_br %71, ^bb18, ^bb19
    ^bb18:  // pred: ^bb17
      %72 = arith.addi %1, %66 : index
      %73 = arith.remsi %72, %c4 : index
      %74 = arith.divsi %72, %c4 : index
      %75 = memref.load %arg7[%74] : memref<?xi32>
      %76 = arith.index_cast %75 : i32 to index
      %77 = memref.load %arg9[%76, %73] : memref<?x4xf32>
      memref.store %77, %arg10[%74, %73] : memref<?x4xf32>
      cf.br ^bb19
    ^bb19:  // 2 preds: ^bb17, ^bb18
      %78 = arith.addi %0, %c42 : index
      %79 = arith.remsi %78, %2 : index
      %c512_18 = arith.constant 512 : index
      %80 = arith.muli %79, %c512_18 : index
      %c-512_19 = arith.constant -512 : index
      %81 = arith.muli %79, %c-512_19 : index
      %82 = arith.addi %81, %dim_1 : index
      %c512_20 = arith.constant 512 : index
      %83 = arith.cmpi slt, %82, %c512_20 : index
      %84 = arith.select %83, %82, %c512_20 : index
      %85 = arith.cmpi slt, %1, %84 : index
      cf.cond_br %85, ^bb20, ^bb21
    ^bb20:  // pred: ^bb19
      %86 = arith.addi %1, %80 : index
      memref.store %c0_i32, %arg11[%86] : memref<?xi32>
      cf.br ^bb21
    ^bb21:  // 2 preds: ^bb19, ^bb20
      %87 = arith.addi %0, %c49 : index
      %88 = arith.remsi %87, %2 : index
      %c512_21 = arith.constant 512 : index
      %89 = arith.muli %88, %c512_21 : index
      %c-512_22 = arith.constant -512 : index
      %90 = arith.muli %88, %c-512_22 : index
      %91 = arith.addi %90, %dim_1 : index
      %c512_23 = arith.constant 512 : index
      %92 = arith.cmpi slt, %91, %c512_23 : index
      %93 = arith.select %92, %91, %c512_23 : index
      %94 = arith.cmpi slt, %1, %93 : index
      cf.cond_br %94, ^bb22, ^bb26
    ^bb22:  // pred: ^bb21
      %95 = arith.addi %1, %89 : index
      cf.br ^bb23(%c0 : index)
    ^bb23(%96: index):  // 2 preds: ^bb22, ^bb24
      %97 = arith.cmpi slt, %96, %c2 : index
      cf.cond_br %97, ^bb24, ^bb25
    ^bb24:  // pred: ^bb23
      %98 = memref.load %arg2[%95, %96] : memref<?x2xi32>
      %99 = memref.load %arg11[%95] : memref<?xi32>
      %100 = arith.addi %99, %98 : i32
      memref.store %100, %arg11[%95] : memref<?xi32>
      %101 = arith.addi %96, %c1 : index
      cf.br ^bb23(%101 : index)
    ^bb25:  // pred: ^bb23
      cf.br ^bb26
    ^bb26:  // 2 preds: ^bb21, ^bb25
      %102 = arith.addi %0, %c56 : index
      %103 = arith.remsi %102, %2 : index
      %c512_24 = arith.constant 512 : index
      %104 = arith.muli %103, %c512_24 : index
      %c-512_25 = arith.constant -512 : index
      %105 = arith.muli %103, %c-512_25 : index
      %106 = arith.addi %105, %arg12 : index
      %c512_26 = arith.constant 512 : index
      %107 = arith.cmpi slt, %106, %c512_26 : index
      %108 = arith.select %107, %106, %c512_26 : index
      %109 = arith.cmpi slt, %1, %108 : index
      cf.cond_br %109, ^bb27, ^bb28
    ^bb27:  // pred: ^bb26
      %110 = arith.addi %1, %104 : index
      %111 = arith.remsi %110, %c4 : index
      %112 = arith.divsi %110, %c4 : index
      %113 = memref.load %arg11[%112] : memref<?xi32>
      %114 = arith.index_cast %113 : i32 to index
      %115 = memref.load %arg13[%114, %111] : memref<?x4xf32>
      memref.store %115, %arg14[%112, %111] : memref<?x4xf32>
      cf.br ^bb28
    ^bb28:  // 2 preds: ^bb26, ^bb27
      gpu.return
    }
  }
}


// -----// IR Dump Before RewriteTFFrameworkAssert (rewrite-tf-framework-assert) //----- //
module attributes {gpu.container_module} {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %3, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %8, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %10, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    tf_framework.assert %arg0, %15, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    tf_framework.assert %arg0, %17, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = arith.cmpi sle, %21, %c0 : index
    %23 = arith.subi %c0, %21 : index
    %24 = arith.subi %21, %c1 : index
    %25 = arith.select %22, %23, %24 : index
    %26 = arith.divsi %25, %c512 : index
    %27 = arith.subi %c0, %26 : index
    %28 = arith.addi %26, %c1 : index
    %29 = arith.select %22, %27, %28 : index
    gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%29, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
  gpu.module @predict_online_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_0_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?x2xi32>, %arg2: memref<?x2xi32>, %arg3: memref<?xi32>, %arg4: index, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>, %arg7: memref<?xi32>, %arg8: index, %arg9: memref<?x4xf32>, %arg10: memref<?x4xf32>, %arg11: memref<?xi32>, %arg12: index, %arg13: memref<?x4xf32>, %arg14: memref<?x4xf32>) kernel {
      %c56 = arith.constant 56 : index
      %c49 = arith.constant 49 : index
      %c42 = arith.constant 42 : index
      %c35 = arith.constant 35 : index
      %c28 = arith.constant 28 : index
      %c21 = arith.constant 21 : index
      %c4 = arith.constant 4 : index
      %c14 = arith.constant 14 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.grid_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
      %dim_0 = memref.dim %arg1, %c0 : memref<?x2xi32>
      %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
      %3 = arith.remsi %0, %2 : index
      %c512 = arith.constant 512 : index
      %4 = arith.muli %3, %c512 : index
      %c-512 = arith.constant -512 : index
      %5 = arith.muli %3, %c-512 : index
      %6 = arith.addi %5, %dim : index
      %c512_2 = arith.constant 512 : index
      %7 = arith.cmpi slt, %6, %c512_2 : index
      %8 = arith.select %7, %6, %c512_2 : index
      %9 = arith.cmpi slt, %1, %8 : index
      cf.cond_br %9, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %10 = arith.addi %1, %4 : index
      memref.store %c0_i32, %arg3[%10] : memref<?xi32>
      cf.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      %11 = arith.addi %0, %c7 : index
      %12 = arith.remsi %11, %2 : index
      %c512_3 = arith.constant 512 : index
      %13 = arith.muli %12, %c512_3 : index
      %c-512_4 = arith.constant -512 : index
      %14 = arith.muli %12, %c-512_4 : index
      %15 = arith.addi %14, %dim : index
      %c512_5 = arith.constant 512 : index
      %16 = arith.cmpi slt, %15, %c512_5 : index
      %17 = arith.select %16, %15, %c512_5 : index
      %18 = arith.cmpi slt, %1, %17 : index
      cf.cond_br %18, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      %19 = arith.addi %1, %13 : index
      cf.br ^bb5(%c0 : index)
    ^bb5(%20: index):  // 2 preds: ^bb4, ^bb6
      %21 = arith.cmpi slt, %20, %c2 : index
      cf.cond_br %21, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %22 = memref.load %arg0[%19, %20] : memref<?x2xi32>
      %23 = memref.load %arg3[%19] : memref<?xi32>
      %24 = arith.addi %23, %22 : i32
      memref.store %24, %arg3[%19] : memref<?xi32>
      %25 = arith.addi %20, %c1 : index
      cf.br ^bb5(%25 : index)
    ^bb7:  // pred: ^bb5
      cf.br ^bb8
    ^bb8:  // 2 preds: ^bb3, ^bb7
      %26 = arith.addi %0, %c14 : index
      %27 = arith.remsi %26, %2 : index
      %c512_6 = arith.constant 512 : index
      %28 = arith.muli %27, %c512_6 : index
      %c-512_7 = arith.constant -512 : index
      %29 = arith.muli %27, %c-512_7 : index
      %30 = arith.addi %29, %arg4 : index
      %c512_8 = arith.constant 512 : index
      %31 = arith.cmpi slt, %30, %c512_8 : index
      %32 = arith.select %31, %30, %c512_8 : index
      %33 = arith.cmpi slt, %1, %32 : index
      cf.cond_br %33, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      %34 = arith.addi %1, %28 : index
      %35 = arith.remsi %34, %c4 : index
      %36 = arith.divsi %34, %c4 : index
      %37 = memref.load %arg3[%36] : memref<?xi32>
      %38 = arith.index_cast %37 : i32 to index
      %39 = memref.load %arg5[%38, %35] : memref<?x4xf32>
      memref.store %39, %arg6[%36, %35] : memref<?x4xf32>
      cf.br ^bb10
    ^bb10:  // 2 preds: ^bb8, ^bb9
      %40 = arith.addi %0, %c21 : index
      %41 = arith.remsi %40, %2 : index
      %c512_9 = arith.constant 512 : index
      %42 = arith.muli %41, %c512_9 : index
      %c-512_10 = arith.constant -512 : index
      %43 = arith.muli %41, %c-512_10 : index
      %44 = arith.addi %43, %dim_0 : index
      %c512_11 = arith.constant 512 : index
      %45 = arith.cmpi slt, %44, %c512_11 : index
      %46 = arith.select %45, %44, %c512_11 : index
      %47 = arith.cmpi slt, %1, %46 : index
      cf.cond_br %47, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      %48 = arith.addi %1, %42 : index
      memref.store %c0_i32, %arg7[%48] : memref<?xi32>
      cf.br ^bb12
    ^bb12:  // 2 preds: ^bb10, ^bb11
      %49 = arith.addi %0, %c28 : index
      %50 = arith.remsi %49, %2 : index
      %c512_12 = arith.constant 512 : index
      %51 = arith.muli %50, %c512_12 : index
      %c-512_13 = arith.constant -512 : index
      %52 = arith.muli %50, %c-512_13 : index
      %53 = arith.addi %52, %dim_0 : index
      %c512_14 = arith.constant 512 : index
      %54 = arith.cmpi slt, %53, %c512_14 : index
      %55 = arith.select %54, %53, %c512_14 : index
      %56 = arith.cmpi slt, %1, %55 : index
      cf.cond_br %56, ^bb13, ^bb17
    ^bb13:  // pred: ^bb12
      %57 = arith.addi %1, %51 : index
      cf.br ^bb14(%c0 : index)
    ^bb14(%58: index):  // 2 preds: ^bb13, ^bb15
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb15, ^bb16
    ^bb15:  // pred: ^bb14
      %60 = memref.load %arg1[%57, %58] : memref<?x2xi32>
      %61 = memref.load %arg7[%57] : memref<?xi32>
      %62 = arith.addi %61, %60 : i32
      memref.store %62, %arg7[%57] : memref<?xi32>
      %63 = arith.addi %58, %c1 : index
      cf.br ^bb14(%63 : index)
    ^bb16:  // pred: ^bb14
      cf.br ^bb17
    ^bb17:  // 2 preds: ^bb12, ^bb16
      %64 = arith.addi %0, %c35 : index
      %65 = arith.remsi %64, %2 : index
      %c512_15 = arith.constant 512 : index
      %66 = arith.muli %65, %c512_15 : index
      %c-512_16 = arith.constant -512 : index
      %67 = arith.muli %65, %c-512_16 : index
      %68 = arith.addi %67, %arg8 : index
      %c512_17 = arith.constant 512 : index
      %69 = arith.cmpi slt, %68, %c512_17 : index
      %70 = arith.select %69, %68, %c512_17 : index
      %71 = arith.cmpi slt, %1, %70 : index
      cf.cond_br %71, ^bb18, ^bb19
    ^bb18:  // pred: ^bb17
      %72 = arith.addi %1, %66 : index
      %73 = arith.remsi %72, %c4 : index
      %74 = arith.divsi %72, %c4 : index
      %75 = memref.load %arg7[%74] : memref<?xi32>
      %76 = arith.index_cast %75 : i32 to index
      %77 = memref.load %arg9[%76, %73] : memref<?x4xf32>
      memref.store %77, %arg10[%74, %73] : memref<?x4xf32>
      cf.br ^bb19
    ^bb19:  // 2 preds: ^bb17, ^bb18
      %78 = arith.addi %0, %c42 : index
      %79 = arith.remsi %78, %2 : index
      %c512_18 = arith.constant 512 : index
      %80 = arith.muli %79, %c512_18 : index
      %c-512_19 = arith.constant -512 : index
      %81 = arith.muli %79, %c-512_19 : index
      %82 = arith.addi %81, %dim_1 : index
      %c512_20 = arith.constant 512 : index
      %83 = arith.cmpi slt, %82, %c512_20 : index
      %84 = arith.select %83, %82, %c512_20 : index
      %85 = arith.cmpi slt, %1, %84 : index
      cf.cond_br %85, ^bb20, ^bb21
    ^bb20:  // pred: ^bb19
      %86 = arith.addi %1, %80 : index
      memref.store %c0_i32, %arg11[%86] : memref<?xi32>
      cf.br ^bb21
    ^bb21:  // 2 preds: ^bb19, ^bb20
      %87 = arith.addi %0, %c49 : index
      %88 = arith.remsi %87, %2 : index
      %c512_21 = arith.constant 512 : index
      %89 = arith.muli %88, %c512_21 : index
      %c-512_22 = arith.constant -512 : index
      %90 = arith.muli %88, %c-512_22 : index
      %91 = arith.addi %90, %dim_1 : index
      %c512_23 = arith.constant 512 : index
      %92 = arith.cmpi slt, %91, %c512_23 : index
      %93 = arith.select %92, %91, %c512_23 : index
      %94 = arith.cmpi slt, %1, %93 : index
      cf.cond_br %94, ^bb22, ^bb26
    ^bb22:  // pred: ^bb21
      %95 = arith.addi %1, %89 : index
      cf.br ^bb23(%c0 : index)
    ^bb23(%96: index):  // 2 preds: ^bb22, ^bb24
      %97 = arith.cmpi slt, %96, %c2 : index
      cf.cond_br %97, ^bb24, ^bb25
    ^bb24:  // pred: ^bb23
      %98 = memref.load %arg2[%95, %96] : memref<?x2xi32>
      %99 = memref.load %arg11[%95] : memref<?xi32>
      %100 = arith.addi %99, %98 : i32
      memref.store %100, %arg11[%95] : memref<?xi32>
      %101 = arith.addi %96, %c1 : index
      cf.br ^bb23(%101 : index)
    ^bb25:  // pred: ^bb23
      cf.br ^bb26
    ^bb26:  // 2 preds: ^bb21, ^bb25
      %102 = arith.addi %0, %c56 : index
      %103 = arith.remsi %102, %2 : index
      %c512_24 = arith.constant 512 : index
      %104 = arith.muli %103, %c512_24 : index
      %c-512_25 = arith.constant -512 : index
      %105 = arith.muli %103, %c-512_25 : index
      %106 = arith.addi %105, %arg12 : index
      %c512_26 = arith.constant 512 : index
      %107 = arith.cmpi slt, %106, %c512_26 : index
      %108 = arith.select %107, %106, %c512_26 : index
      %109 = arith.cmpi slt, %1, %108 : index
      cf.cond_br %109, ^bb27, ^bb28
    ^bb27:  // pred: ^bb26
      %110 = arith.addi %1, %104 : index
      %111 = arith.remsi %110, %c4 : index
      %112 = arith.divsi %110, %c4 : index
      %113 = memref.load %arg11[%112] : memref<?xi32>
      %114 = arith.index_cast %113 : i32 to index
      %115 = memref.load %arg13[%114, %111] : memref<?x4xf32>
      memref.store %115, %arg14[%112, %111] : memref<?x4xf32>
      cf.br ^bb28
    ^bb28:  // 2 preds: ^bb26, ^bb27
      gpu.return
    }
  }
}


// -----// IR Dump After RewriteTFFrameworkAssert (rewrite-tf-framework-assert) //----- //
module attributes {gpu.container_module} {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    cf.cond_br %1, ^bb1, ^bb7
  ^bb1:  // pred: ^bb0
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    cf.cond_br %3, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    cf.cond_br %8, ^bb3, ^bb9
  ^bb3:  // pred: ^bb2
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    cf.cond_br %10, ^bb4, ^bb10
  ^bb4:  // pred: ^bb3
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    cf.cond_br %15, ^bb5, ^bb11
  ^bb5:  // pred: ^bb4
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    cf.cond_br %17, ^bb6, ^bb12
  ^bb6:  // pred: ^bb5
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = arith.cmpi sle, %21, %c0 : index
    %23 = arith.subi %c0, %21 : index
    %24 = arith.subi %21, %c1 : index
    %25 = arith.select %22, %23, %24 : index
    %26 = arith.divsi %25, %c512 : index
    %27 = arith.subi %c0, %26 : index
    %28 = arith.addi %26, %c1 : index
    %29 = arith.select %22, %27, %28 : index
    gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%29, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb7:  // pred: ^bb0
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %30 = tf_framework.null_memref : memref<?x4xf32>
    %31 = tf_framework.null_memref : memref<?x4xf32>
    %32 = tf_framework.null_memref : memref<?x4xf32>
    return %30, %31, %32 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb8:  // pred: ^bb1
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %33 = tf_framework.null_memref : memref<?x4xf32>
    %34 = tf_framework.null_memref : memref<?x4xf32>
    %35 = tf_framework.null_memref : memref<?x4xf32>
    return %33, %34, %35 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb9:  // pred: ^bb2
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %36 = tf_framework.null_memref : memref<?x4xf32>
    %37 = tf_framework.null_memref : memref<?x4xf32>
    %38 = tf_framework.null_memref : memref<?x4xf32>
    return %36, %37, %38 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb10:  // pred: ^bb3
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %39 = tf_framework.null_memref : memref<?x4xf32>
    %40 = tf_framework.null_memref : memref<?x4xf32>
    %41 = tf_framework.null_memref : memref<?x4xf32>
    return %39, %40, %41 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb11:  // pred: ^bb4
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %42 = tf_framework.null_memref : memref<?x4xf32>
    %43 = tf_framework.null_memref : memref<?x4xf32>
    %44 = tf_framework.null_memref : memref<?x4xf32>
    return %42, %43, %44 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb12:  // pred: ^bb5
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %45 = tf_framework.null_memref : memref<?x4xf32>
    %46 = tf_framework.null_memref : memref<?x4xf32>
    %47 = tf_framework.null_memref : memref<?x4xf32>
    return %45, %46, %47 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
  gpu.module @predict_online_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_0_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?x2xi32>, %arg2: memref<?x2xi32>, %arg3: memref<?xi32>, %arg4: index, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>, %arg7: memref<?xi32>, %arg8: index, %arg9: memref<?x4xf32>, %arg10: memref<?x4xf32>, %arg11: memref<?xi32>, %arg12: index, %arg13: memref<?x4xf32>, %arg14: memref<?x4xf32>) kernel {
      %c56 = arith.constant 56 : index
      %c49 = arith.constant 49 : index
      %c42 = arith.constant 42 : index
      %c35 = arith.constant 35 : index
      %c28 = arith.constant 28 : index
      %c21 = arith.constant 21 : index
      %c4 = arith.constant 4 : index
      %c14 = arith.constant 14 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.grid_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
      %dim_0 = memref.dim %arg1, %c0 : memref<?x2xi32>
      %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
      %3 = arith.remsi %0, %2 : index
      %c512 = arith.constant 512 : index
      %4 = arith.muli %3, %c512 : index
      %c-512 = arith.constant -512 : index
      %5 = arith.muli %3, %c-512 : index
      %6 = arith.addi %5, %dim : index
      %c512_2 = arith.constant 512 : index
      %7 = arith.cmpi slt, %6, %c512_2 : index
      %8 = arith.select %7, %6, %c512_2 : index
      %9 = arith.cmpi slt, %1, %8 : index
      cf.cond_br %9, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %10 = arith.addi %1, %4 : index
      memref.store %c0_i32, %arg3[%10] : memref<?xi32>
      cf.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      %11 = arith.addi %0, %c7 : index
      %12 = arith.remsi %11, %2 : index
      %c512_3 = arith.constant 512 : index
      %13 = arith.muli %12, %c512_3 : index
      %c-512_4 = arith.constant -512 : index
      %14 = arith.muli %12, %c-512_4 : index
      %15 = arith.addi %14, %dim : index
      %c512_5 = arith.constant 512 : index
      %16 = arith.cmpi slt, %15, %c512_5 : index
      %17 = arith.select %16, %15, %c512_5 : index
      %18 = arith.cmpi slt, %1, %17 : index
      cf.cond_br %18, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      %19 = arith.addi %1, %13 : index
      cf.br ^bb5(%c0 : index)
    ^bb5(%20: index):  // 2 preds: ^bb4, ^bb6
      %21 = arith.cmpi slt, %20, %c2 : index
      cf.cond_br %21, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %22 = memref.load %arg0[%19, %20] : memref<?x2xi32>
      %23 = memref.load %arg3[%19] : memref<?xi32>
      %24 = arith.addi %23, %22 : i32
      memref.store %24, %arg3[%19] : memref<?xi32>
      %25 = arith.addi %20, %c1 : index
      cf.br ^bb5(%25 : index)
    ^bb7:  // pred: ^bb5
      cf.br ^bb8
    ^bb8:  // 2 preds: ^bb3, ^bb7
      %26 = arith.addi %0, %c14 : index
      %27 = arith.remsi %26, %2 : index
      %c512_6 = arith.constant 512 : index
      %28 = arith.muli %27, %c512_6 : index
      %c-512_7 = arith.constant -512 : index
      %29 = arith.muli %27, %c-512_7 : index
      %30 = arith.addi %29, %arg4 : index
      %c512_8 = arith.constant 512 : index
      %31 = arith.cmpi slt, %30, %c512_8 : index
      %32 = arith.select %31, %30, %c512_8 : index
      %33 = arith.cmpi slt, %1, %32 : index
      cf.cond_br %33, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      %34 = arith.addi %1, %28 : index
      %35 = arith.remsi %34, %c4 : index
      %36 = arith.divsi %34, %c4 : index
      %37 = memref.load %arg3[%36] : memref<?xi32>
      %38 = arith.index_cast %37 : i32 to index
      %39 = memref.load %arg5[%38, %35] : memref<?x4xf32>
      memref.store %39, %arg6[%36, %35] : memref<?x4xf32>
      cf.br ^bb10
    ^bb10:  // 2 preds: ^bb8, ^bb9
      %40 = arith.addi %0, %c21 : index
      %41 = arith.remsi %40, %2 : index
      %c512_9 = arith.constant 512 : index
      %42 = arith.muli %41, %c512_9 : index
      %c-512_10 = arith.constant -512 : index
      %43 = arith.muli %41, %c-512_10 : index
      %44 = arith.addi %43, %dim_0 : index
      %c512_11 = arith.constant 512 : index
      %45 = arith.cmpi slt, %44, %c512_11 : index
      %46 = arith.select %45, %44, %c512_11 : index
      %47 = arith.cmpi slt, %1, %46 : index
      cf.cond_br %47, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      %48 = arith.addi %1, %42 : index
      memref.store %c0_i32, %arg7[%48] : memref<?xi32>
      cf.br ^bb12
    ^bb12:  // 2 preds: ^bb10, ^bb11
      %49 = arith.addi %0, %c28 : index
      %50 = arith.remsi %49, %2 : index
      %c512_12 = arith.constant 512 : index
      %51 = arith.muli %50, %c512_12 : index
      %c-512_13 = arith.constant -512 : index
      %52 = arith.muli %50, %c-512_13 : index
      %53 = arith.addi %52, %dim_0 : index
      %c512_14 = arith.constant 512 : index
      %54 = arith.cmpi slt, %53, %c512_14 : index
      %55 = arith.select %54, %53, %c512_14 : index
      %56 = arith.cmpi slt, %1, %55 : index
      cf.cond_br %56, ^bb13, ^bb17
    ^bb13:  // pred: ^bb12
      %57 = arith.addi %1, %51 : index
      cf.br ^bb14(%c0 : index)
    ^bb14(%58: index):  // 2 preds: ^bb13, ^bb15
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb15, ^bb16
    ^bb15:  // pred: ^bb14
      %60 = memref.load %arg1[%57, %58] : memref<?x2xi32>
      %61 = memref.load %arg7[%57] : memref<?xi32>
      %62 = arith.addi %61, %60 : i32
      memref.store %62, %arg7[%57] : memref<?xi32>
      %63 = arith.addi %58, %c1 : index
      cf.br ^bb14(%63 : index)
    ^bb16:  // pred: ^bb14
      cf.br ^bb17
    ^bb17:  // 2 preds: ^bb12, ^bb16
      %64 = arith.addi %0, %c35 : index
      %65 = arith.remsi %64, %2 : index
      %c512_15 = arith.constant 512 : index
      %66 = arith.muli %65, %c512_15 : index
      %c-512_16 = arith.constant -512 : index
      %67 = arith.muli %65, %c-512_16 : index
      %68 = arith.addi %67, %arg8 : index
      %c512_17 = arith.constant 512 : index
      %69 = arith.cmpi slt, %68, %c512_17 : index
      %70 = arith.select %69, %68, %c512_17 : index
      %71 = arith.cmpi slt, %1, %70 : index
      cf.cond_br %71, ^bb18, ^bb19
    ^bb18:  // pred: ^bb17
      %72 = arith.addi %1, %66 : index
      %73 = arith.remsi %72, %c4 : index
      %74 = arith.divsi %72, %c4 : index
      %75 = memref.load %arg7[%74] : memref<?xi32>
      %76 = arith.index_cast %75 : i32 to index
      %77 = memref.load %arg9[%76, %73] : memref<?x4xf32>
      memref.store %77, %arg10[%74, %73] : memref<?x4xf32>
      cf.br ^bb19
    ^bb19:  // 2 preds: ^bb17, ^bb18
      %78 = arith.addi %0, %c42 : index
      %79 = arith.remsi %78, %2 : index
      %c512_18 = arith.constant 512 : index
      %80 = arith.muli %79, %c512_18 : index
      %c-512_19 = arith.constant -512 : index
      %81 = arith.muli %79, %c-512_19 : index
      %82 = arith.addi %81, %dim_1 : index
      %c512_20 = arith.constant 512 : index
      %83 = arith.cmpi slt, %82, %c512_20 : index
      %84 = arith.select %83, %82, %c512_20 : index
      %85 = arith.cmpi slt, %1, %84 : index
      cf.cond_br %85, ^bb20, ^bb21
    ^bb20:  // pred: ^bb19
      %86 = arith.addi %1, %80 : index
      memref.store %c0_i32, %arg11[%86] : memref<?xi32>
      cf.br ^bb21
    ^bb21:  // 2 preds: ^bb19, ^bb20
      %87 = arith.addi %0, %c49 : index
      %88 = arith.remsi %87, %2 : index
      %c512_21 = arith.constant 512 : index
      %89 = arith.muli %88, %c512_21 : index
      %c-512_22 = arith.constant -512 : index
      %90 = arith.muli %88, %c-512_22 : index
      %91 = arith.addi %90, %dim_1 : index
      %c512_23 = arith.constant 512 : index
      %92 = arith.cmpi slt, %91, %c512_23 : index
      %93 = arith.select %92, %91, %c512_23 : index
      %94 = arith.cmpi slt, %1, %93 : index
      cf.cond_br %94, ^bb22, ^bb26
    ^bb22:  // pred: ^bb21
      %95 = arith.addi %1, %89 : index
      cf.br ^bb23(%c0 : index)
    ^bb23(%96: index):  // 2 preds: ^bb22, ^bb24
      %97 = arith.cmpi slt, %96, %c2 : index
      cf.cond_br %97, ^bb24, ^bb25
    ^bb24:  // pred: ^bb23
      %98 = memref.load %arg2[%95, %96] : memref<?x2xi32>
      %99 = memref.load %arg11[%95] : memref<?xi32>
      %100 = arith.addi %99, %98 : i32
      memref.store %100, %arg11[%95] : memref<?xi32>
      %101 = arith.addi %96, %c1 : index
      cf.br ^bb23(%101 : index)
    ^bb25:  // pred: ^bb23
      cf.br ^bb26
    ^bb26:  // 2 preds: ^bb21, ^bb25
      %102 = arith.addi %0, %c56 : index
      %103 = arith.remsi %102, %2 : index
      %c512_24 = arith.constant 512 : index
      %104 = arith.muli %103, %c512_24 : index
      %c-512_25 = arith.constant -512 : index
      %105 = arith.muli %103, %c-512_25 : index
      %106 = arith.addi %105, %arg12 : index
      %c512_26 = arith.constant 512 : index
      %107 = arith.cmpi slt, %106, %c512_26 : index
      %108 = arith.select %107, %106, %c512_26 : index
      %109 = arith.cmpi slt, %1, %108 : index
      cf.cond_br %109, ^bb27, ^bb28
    ^bb27:  // pred: ^bb26
      %110 = arith.addi %1, %104 : index
      %111 = arith.remsi %110, %c4 : index
      %112 = arith.divsi %110, %c4 : index
      %113 = memref.load %arg11[%112] : memref<?xi32>
      %114 = arith.index_cast %113 : i32 to index
      %115 = memref.load %arg13[%114, %111] : memref<?x4xf32>
      memref.store %115, %arg14[%112, %111] : memref<?x4xf32>
      cf.br ^bb28
    ^bb28:  // 2 preds: ^bb26, ^bb27
      gpu.return
    }
  }
}


// -----// IR Dump Before InterleaveLoadAndCompute (interleave-load-and-compute) //----- //
module attributes {gpu.container_module} {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    cf.cond_br %1, ^bb1, ^bb7
  ^bb1:  // pred: ^bb0
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    cf.cond_br %3, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    cf.cond_br %8, ^bb3, ^bb9
  ^bb3:  // pred: ^bb2
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    cf.cond_br %10, ^bb4, ^bb10
  ^bb4:  // pred: ^bb3
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    cf.cond_br %15, ^bb5, ^bb11
  ^bb5:  // pred: ^bb4
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    cf.cond_br %17, ^bb6, ^bb12
  ^bb6:  // pred: ^bb5
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = arith.cmpi sle, %21, %c0 : index
    %23 = arith.subi %c0, %21 : index
    %24 = arith.subi %21, %c1 : index
    %25 = arith.select %22, %23, %24 : index
    %26 = arith.divsi %25, %c512 : index
    %27 = arith.subi %c0, %26 : index
    %28 = arith.addi %26, %c1 : index
    %29 = arith.select %22, %27, %28 : index
    gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%29, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb7:  // pred: ^bb0
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %30 = tf_framework.null_memref : memref<?x4xf32>
    %31 = tf_framework.null_memref : memref<?x4xf32>
    %32 = tf_framework.null_memref : memref<?x4xf32>
    return %30, %31, %32 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb8:  // pred: ^bb1
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %33 = tf_framework.null_memref : memref<?x4xf32>
    %34 = tf_framework.null_memref : memref<?x4xf32>
    %35 = tf_framework.null_memref : memref<?x4xf32>
    return %33, %34, %35 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb9:  // pred: ^bb2
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %36 = tf_framework.null_memref : memref<?x4xf32>
    %37 = tf_framework.null_memref : memref<?x4xf32>
    %38 = tf_framework.null_memref : memref<?x4xf32>
    return %36, %37, %38 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb10:  // pred: ^bb3
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %39 = tf_framework.null_memref : memref<?x4xf32>
    %40 = tf_framework.null_memref : memref<?x4xf32>
    %41 = tf_framework.null_memref : memref<?x4xf32>
    return %39, %40, %41 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb11:  // pred: ^bb4
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %42 = tf_framework.null_memref : memref<?x4xf32>
    %43 = tf_framework.null_memref : memref<?x4xf32>
    %44 = tf_framework.null_memref : memref<?x4xf32>
    return %42, %43, %44 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb12:  // pred: ^bb5
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %45 = tf_framework.null_memref : memref<?x4xf32>
    %46 = tf_framework.null_memref : memref<?x4xf32>
    %47 = tf_framework.null_memref : memref<?x4xf32>
    return %45, %46, %47 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
  gpu.module @predict_online_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_0_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?x2xi32>, %arg2: memref<?x2xi32>, %arg3: memref<?xi32>, %arg4: index, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>, %arg7: memref<?xi32>, %arg8: index, %arg9: memref<?x4xf32>, %arg10: memref<?x4xf32>, %arg11: memref<?xi32>, %arg12: index, %arg13: memref<?x4xf32>, %arg14: memref<?x4xf32>) kernel {
      %c56 = arith.constant 56 : index
      %c49 = arith.constant 49 : index
      %c42 = arith.constant 42 : index
      %c35 = arith.constant 35 : index
      %c28 = arith.constant 28 : index
      %c21 = arith.constant 21 : index
      %c4 = arith.constant 4 : index
      %c14 = arith.constant 14 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.grid_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
      %dim_0 = memref.dim %arg1, %c0 : memref<?x2xi32>
      %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
      %3 = arith.remsi %0, %2 : index
      %c512 = arith.constant 512 : index
      %4 = arith.muli %3, %c512 : index
      %c-512 = arith.constant -512 : index
      %5 = arith.muli %3, %c-512 : index
      %6 = arith.addi %5, %dim : index
      %c512_2 = arith.constant 512 : index
      %7 = arith.cmpi slt, %6, %c512_2 : index
      %8 = arith.select %7, %6, %c512_2 : index
      %9 = arith.cmpi slt, %1, %8 : index
      cf.cond_br %9, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %10 = arith.addi %1, %4 : index
      memref.store %c0_i32, %arg3[%10] : memref<?xi32>
      cf.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      %11 = arith.addi %0, %c7 : index
      %12 = arith.remsi %11, %2 : index
      %c512_3 = arith.constant 512 : index
      %13 = arith.muli %12, %c512_3 : index
      %c-512_4 = arith.constant -512 : index
      %14 = arith.muli %12, %c-512_4 : index
      %15 = arith.addi %14, %dim : index
      %c512_5 = arith.constant 512 : index
      %16 = arith.cmpi slt, %15, %c512_5 : index
      %17 = arith.select %16, %15, %c512_5 : index
      %18 = arith.cmpi slt, %1, %17 : index
      cf.cond_br %18, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      %19 = arith.addi %1, %13 : index
      cf.br ^bb5(%c0 : index)
    ^bb5(%20: index):  // 2 preds: ^bb4, ^bb6
      %21 = arith.cmpi slt, %20, %c2 : index
      cf.cond_br %21, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %22 = memref.load %arg0[%19, %20] : memref<?x2xi32>
      %23 = memref.load %arg3[%19] : memref<?xi32>
      %24 = arith.addi %23, %22 : i32
      memref.store %24, %arg3[%19] : memref<?xi32>
      %25 = arith.addi %20, %c1 : index
      cf.br ^bb5(%25 : index)
    ^bb7:  // pred: ^bb5
      cf.br ^bb8
    ^bb8:  // 2 preds: ^bb3, ^bb7
      %26 = arith.addi %0, %c14 : index
      %27 = arith.remsi %26, %2 : index
      %c512_6 = arith.constant 512 : index
      %28 = arith.muli %27, %c512_6 : index
      %c-512_7 = arith.constant -512 : index
      %29 = arith.muli %27, %c-512_7 : index
      %30 = arith.addi %29, %arg4 : index
      %c512_8 = arith.constant 512 : index
      %31 = arith.cmpi slt, %30, %c512_8 : index
      %32 = arith.select %31, %30, %c512_8 : index
      %33 = arith.cmpi slt, %1, %32 : index
      cf.cond_br %33, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      %34 = arith.addi %1, %28 : index
      %35 = arith.remsi %34, %c4 : index
      %36 = arith.divsi %34, %c4 : index
      %37 = memref.load %arg3[%36] : memref<?xi32>
      %38 = arith.index_cast %37 : i32 to index
      %39 = memref.load %arg5[%38, %35] : memref<?x4xf32>
      memref.store %39, %arg6[%36, %35] : memref<?x4xf32>
      cf.br ^bb10
    ^bb10:  // 2 preds: ^bb8, ^bb9
      %40 = arith.addi %0, %c21 : index
      %41 = arith.remsi %40, %2 : index
      %c512_9 = arith.constant 512 : index
      %42 = arith.muli %41, %c512_9 : index
      %c-512_10 = arith.constant -512 : index
      %43 = arith.muli %41, %c-512_10 : index
      %44 = arith.addi %43, %dim_0 : index
      %c512_11 = arith.constant 512 : index
      %45 = arith.cmpi slt, %44, %c512_11 : index
      %46 = arith.select %45, %44, %c512_11 : index
      %47 = arith.cmpi slt, %1, %46 : index
      cf.cond_br %47, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      %48 = arith.addi %1, %42 : index
      memref.store %c0_i32, %arg7[%48] : memref<?xi32>
      cf.br ^bb12
    ^bb12:  // 2 preds: ^bb10, ^bb11
      %49 = arith.addi %0, %c28 : index
      %50 = arith.remsi %49, %2 : index
      %c512_12 = arith.constant 512 : index
      %51 = arith.muli %50, %c512_12 : index
      %c-512_13 = arith.constant -512 : index
      %52 = arith.muli %50, %c-512_13 : index
      %53 = arith.addi %52, %dim_0 : index
      %c512_14 = arith.constant 512 : index
      %54 = arith.cmpi slt, %53, %c512_14 : index
      %55 = arith.select %54, %53, %c512_14 : index
      %56 = arith.cmpi slt, %1, %55 : index
      cf.cond_br %56, ^bb13, ^bb17
    ^bb13:  // pred: ^bb12
      %57 = arith.addi %1, %51 : index
      cf.br ^bb14(%c0 : index)
    ^bb14(%58: index):  // 2 preds: ^bb13, ^bb15
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb15, ^bb16
    ^bb15:  // pred: ^bb14
      %60 = memref.load %arg1[%57, %58] : memref<?x2xi32>
      %61 = memref.load %arg7[%57] : memref<?xi32>
      %62 = arith.addi %61, %60 : i32
      memref.store %62, %arg7[%57] : memref<?xi32>
      %63 = arith.addi %58, %c1 : index
      cf.br ^bb14(%63 : index)
    ^bb16:  // pred: ^bb14
      cf.br ^bb17
    ^bb17:  // 2 preds: ^bb12, ^bb16
      %64 = arith.addi %0, %c35 : index
      %65 = arith.remsi %64, %2 : index
      %c512_15 = arith.constant 512 : index
      %66 = arith.muli %65, %c512_15 : index
      %c-512_16 = arith.constant -512 : index
      %67 = arith.muli %65, %c-512_16 : index
      %68 = arith.addi %67, %arg8 : index
      %c512_17 = arith.constant 512 : index
      %69 = arith.cmpi slt, %68, %c512_17 : index
      %70 = arith.select %69, %68, %c512_17 : index
      %71 = arith.cmpi slt, %1, %70 : index
      cf.cond_br %71, ^bb18, ^bb19
    ^bb18:  // pred: ^bb17
      %72 = arith.addi %1, %66 : index
      %73 = arith.remsi %72, %c4 : index
      %74 = arith.divsi %72, %c4 : index
      %75 = memref.load %arg7[%74] : memref<?xi32>
      %76 = arith.index_cast %75 : i32 to index
      %77 = memref.load %arg9[%76, %73] : memref<?x4xf32>
      memref.store %77, %arg10[%74, %73] : memref<?x4xf32>
      cf.br ^bb19
    ^bb19:  // 2 preds: ^bb17, ^bb18
      %78 = arith.addi %0, %c42 : index
      %79 = arith.remsi %78, %2 : index
      %c512_18 = arith.constant 512 : index
      %80 = arith.muli %79, %c512_18 : index
      %c-512_19 = arith.constant -512 : index
      %81 = arith.muli %79, %c-512_19 : index
      %82 = arith.addi %81, %dim_1 : index
      %c512_20 = arith.constant 512 : index
      %83 = arith.cmpi slt, %82, %c512_20 : index
      %84 = arith.select %83, %82, %c512_20 : index
      %85 = arith.cmpi slt, %1, %84 : index
      cf.cond_br %85, ^bb20, ^bb21
    ^bb20:  // pred: ^bb19
      %86 = arith.addi %1, %80 : index
      memref.store %c0_i32, %arg11[%86] : memref<?xi32>
      cf.br ^bb21
    ^bb21:  // 2 preds: ^bb19, ^bb20
      %87 = arith.addi %0, %c49 : index
      %88 = arith.remsi %87, %2 : index
      %c512_21 = arith.constant 512 : index
      %89 = arith.muli %88, %c512_21 : index
      %c-512_22 = arith.constant -512 : index
      %90 = arith.muli %88, %c-512_22 : index
      %91 = arith.addi %90, %dim_1 : index
      %c512_23 = arith.constant 512 : index
      %92 = arith.cmpi slt, %91, %c512_23 : index
      %93 = arith.select %92, %91, %c512_23 : index
      %94 = arith.cmpi slt, %1, %93 : index
      cf.cond_br %94, ^bb22, ^bb26
    ^bb22:  // pred: ^bb21
      %95 = arith.addi %1, %89 : index
      cf.br ^bb23(%c0 : index)
    ^bb23(%96: index):  // 2 preds: ^bb22, ^bb24
      %97 = arith.cmpi slt, %96, %c2 : index
      cf.cond_br %97, ^bb24, ^bb25
    ^bb24:  // pred: ^bb23
      %98 = memref.load %arg2[%95, %96] : memref<?x2xi32>
      %99 = memref.load %arg11[%95] : memref<?xi32>
      %100 = arith.addi %99, %98 : i32
      memref.store %100, %arg11[%95] : memref<?xi32>
      %101 = arith.addi %96, %c1 : index
      cf.br ^bb23(%101 : index)
    ^bb25:  // pred: ^bb23
      cf.br ^bb26
    ^bb26:  // 2 preds: ^bb21, ^bb25
      %102 = arith.addi %0, %c56 : index
      %103 = arith.remsi %102, %2 : index
      %c512_24 = arith.constant 512 : index
      %104 = arith.muli %103, %c512_24 : index
      %c-512_25 = arith.constant -512 : index
      %105 = arith.muli %103, %c-512_25 : index
      %106 = arith.addi %105, %arg12 : index
      %c512_26 = arith.constant 512 : index
      %107 = arith.cmpi slt, %106, %c512_26 : index
      %108 = arith.select %107, %106, %c512_26 : index
      %109 = arith.cmpi slt, %1, %108 : index
      cf.cond_br %109, ^bb27, ^bb28
    ^bb27:  // pred: ^bb26
      %110 = arith.addi %1, %104 : index
      %111 = arith.remsi %110, %c4 : index
      %112 = arith.divsi %110, %c4 : index
      %113 = memref.load %arg11[%112] : memref<?xi32>
      %114 = arith.index_cast %113 : i32 to index
      %115 = memref.load %arg13[%114, %111] : memref<?x4xf32>
      memref.store %115, %arg14[%112, %111] : memref<?x4xf32>
      cf.br ^bb28
    ^bb28:  // 2 preds: ^bb26, ^bb27
      gpu.return
    }
  }
}


// -----// IR Dump After InterleaveLoadAndCompute (interleave-load-and-compute) //----- //
module attributes {gpu.container_module} {
  func.func @predict_online_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}, %arg6: memref<?x4xf32>) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c40960 = arith.constant 40960 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dim = memref.dim %arg1, %c0 : memref<?x2xi32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?xi32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?xi32> -> i1
    cf.cond_br %1, ^bb1, ^bb7
  ^bb1:  // pred: ^bb0
    %2 = tf_framework.alloc(%arg0, %dim) : memref<?x4xf32>
    %3 = tf_framework.is_valid_memref(%2) : memref<?x4xf32> -> i1
    cf.cond_br %3, ^bb2, ^bb8
  ^bb2:  // pred: ^bb1
    %4 = arith.muli %dim, %c4 : index
    %5 = arith.maxui %dim, %4 : index
    %dim_0 = memref.dim %arg3, %c0 : memref<?x2xi32>
    %6 = arith.maxui %5, %dim_0 : index
    %7 = tf_framework.alloc(%arg0, %dim_0) : memref<?xi32>
    %8 = tf_framework.is_valid_memref(%7) : memref<?xi32> -> i1
    cf.cond_br %8, ^bb3, ^bb9
  ^bb3:  // pred: ^bb2
    %9 = tf_framework.alloc(%arg0, %dim_0) : memref<?x4xf32>
    %10 = tf_framework.is_valid_memref(%9) : memref<?x4xf32> -> i1
    cf.cond_br %10, ^bb4, ^bb10
  ^bb4:  // pred: ^bb3
    %11 = arith.muli %dim_0, %c4 : index
    %12 = arith.maxui %6, %11 : index
    %dim_1 = memref.dim %arg5, %c0 : memref<?x2xi32>
    %13 = arith.maxui %12, %dim_1 : index
    %14 = tf_framework.alloc(%arg0, %dim_1) : memref<?xi32>
    %15 = tf_framework.is_valid_memref(%14) : memref<?xi32> -> i1
    cf.cond_br %15, ^bb5, ^bb11
  ^bb5:  // pred: ^bb4
    %16 = tf_framework.alloc(%arg0, %dim_1) : memref<?x4xf32>
    %17 = tf_framework.is_valid_memref(%16) : memref<?x4xf32> -> i1
    cf.cond_br %17, ^bb6, ^bb12
  ^bb6:  // pred: ^bb5
    %18 = arith.muli %dim_1, %c4 : index
    %19 = arith.maxui %13, %18 : index
    %20 = arith.cmpi ugt, %19, %c40960 : index
    %21 = arith.select %20, %19, %c40960 : index
    %22 = arith.cmpi sle, %21, %c0 : index
    %23 = arith.subi %c0, %21 : index
    %24 = arith.subi %21, %c1 : index
    %25 = arith.select %22, %23, %24 : index
    %26 = arith.divsi %25, %c512 : index
    %27 = arith.subi %c0, %26 : index
    %28 = arith.addi %26, %c1 : index
    %29 = arith.select %22, %27, %28 : index
    gpu.launch_func  @predict_online_0_kernel::@predict_online_0_kernel blocks in (%29, %c1, %c1) threads in (%c512, %c1, %c1) args(%arg1 : memref<?x2xi32>, %arg3 : memref<?x2xi32>, %arg5 : memref<?x2xi32>, %0 : memref<?xi32>, %4 : index, %arg2 : memref<?x4xf32>, %2 : memref<?x4xf32>, %7 : memref<?xi32>, %11 : index, %arg4 : memref<?x4xf32>, %9 : memref<?x4xf32>, %14 : memref<?xi32>, %18 : index, %arg6 : memref<?x4xf32>, %16 : memref<?x4xf32>)
    tf_framework.dealloc(%arg0, %14) : memref<?xi32>
    tf_framework.dealloc(%arg0, %7) : memref<?xi32>
    tf_framework.dealloc(%arg0, %0) : memref<?xi32>
    return %2, %9, %16 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb7:  // pred: ^bb0
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %30 = tf_framework.null_memref : memref<?x4xf32>
    %31 = tf_framework.null_memref : memref<?x4xf32>
    %32 = tf_framework.null_memref : memref<?x4xf32>
    return %30, %31, %32 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb8:  // pred: ^bb1
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %33 = tf_framework.null_memref : memref<?x4xf32>
    %34 = tf_framework.null_memref : memref<?x4xf32>
    %35 = tf_framework.null_memref : memref<?x4xf32>
    return %33, %34, %35 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb9:  // pred: ^bb2
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %36 = tf_framework.null_memref : memref<?x4xf32>
    %37 = tf_framework.null_memref : memref<?x4xf32>
    %38 = tf_framework.null_memref : memref<?x4xf32>
    return %36, %37, %38 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb10:  // pred: ^bb3
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %39 = tf_framework.null_memref : memref<?x4xf32>
    %40 = tf_framework.null_memref : memref<?x4xf32>
    %41 = tf_framework.null_memref : memref<?x4xf32>
    return %39, %40, %41 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb11:  // pred: ^bb4
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %42 = tf_framework.null_memref : memref<?x4xf32>
    %43 = tf_framework.null_memref : memref<?x4xf32>
    %44 = tf_framework.null_memref : memref<?x4xf32>
    return %42, %43, %44 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  ^bb12:  // pred: ^bb5
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %45 = tf_framework.null_memref : memref<?x4xf32>
    %46 = tf_framework.null_memref : memref<?x4xf32>
    %47 = tf_framework.null_memref : memref<?x4xf32>
    return %45, %46, %47 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
  gpu.module @predict_online_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_0_kernel(%arg0: memref<?x2xi32>, %arg1: memref<?x2xi32>, %arg2: memref<?x2xi32>, %arg3: memref<?xi32>, %arg4: index, %arg5: memref<?x4xf32>, %arg6: memref<?x4xf32>, %arg7: memref<?xi32>, %arg8: index, %arg9: memref<?x4xf32>, %arg10: memref<?x4xf32>, %arg11: memref<?xi32>, %arg12: index, %arg13: memref<?x4xf32>, %arg14: memref<?x4xf32>) kernel {
      %c56 = arith.constant 56 : index
      %c49 = arith.constant 49 : index
      %c42 = arith.constant 42 : index
      %c35 = arith.constant 35 : index
      %c28 = arith.constant 28 : index
      %c21 = arith.constant 21 : index
      %c4 = arith.constant 4 : index
      %c14 = arith.constant 14 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.grid_dim  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
      %dim_0 = memref.dim %arg1, %c0 : memref<?x2xi32>
      %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
      %3 = arith.remsi %0, %2 : index
      %c512 = arith.constant 512 : index
      %4 = arith.muli %3, %c512 : index
      %c-512 = arith.constant -512 : index
      %5 = arith.muli %3, %c-512 : index
      %6 = arith.addi %5, %dim : index
      %c512_2 = arith.constant 512 : index
      %7 = arith.cmpi slt, %6, %c512_2 : index
      %8 = arith.select %7, %6, %c512_2 : index
      %9 = arith.cmpi slt, %1, %8 : index
      cf.cond_br %9, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %10 = arith.addi %1, %4 : index
      memref.store %c0_i32, %arg3[%10] : memref<?xi32>
      cf.br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      %11 = arith.addi %0, %c7 : index
      %12 = arith.remsi %11, %2 : index
      %c512_3 = arith.constant 512 : index
      %13 = arith.muli %12, %c512_3 : index
      %c-512_4 = arith.constant -512 : index
      %14 = arith.muli %12, %c-512_4 : index
      %15 = arith.addi %14, %dim : index
      %c512_5 = arith.constant 512 : index
      %16 = arith.cmpi slt, %15, %c512_5 : index
      %17 = arith.select %16, %15, %c512_5 : index
      %18 = arith.cmpi slt, %1, %17 : index
      cf.cond_br %18, ^bb4, ^bb8
    ^bb4:  // pred: ^bb3
      %19 = arith.addi %1, %13 : index
      cf.br ^bb5(%c0 : index)
    ^bb5(%20: index):  // 2 preds: ^bb4, ^bb6
      %21 = arith.cmpi slt, %20, %c2 : index
      cf.cond_br %21, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %22 = memref.load %arg0[%19, %20] : memref<?x2xi32>
      %23 = memref.load %arg3[%19] : memref<?xi32>
      %24 = arith.addi %23, %22 : i32
      memref.store %24, %arg3[%19] : memref<?xi32>
      %25 = arith.addi %20, %c1 : index
      cf.br ^bb5(%25 : index)
    ^bb7:  // pred: ^bb5
      cf.br ^bb8
    ^bb8:  // 2 preds: ^bb3, ^bb7
      %26 = arith.addi %0, %c14 : index
      %27 = arith.remsi %26, %2 : index
      %c512_6 = arith.constant 512 : index
      %28 = arith.muli %27, %c512_6 : index
      %c-512_7 = arith.constant -512 : index
      %29 = arith.muli %27, %c-512_7 : index
      %30 = arith.addi %29, %arg4 : index
      %c512_8 = arith.constant 512 : index
      %31 = arith.cmpi slt, %30, %c512_8 : index
      %32 = arith.select %31, %30, %c512_8 : index
      %33 = arith.cmpi slt, %1, %32 : index
      cf.cond_br %33, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      %34 = arith.addi %1, %28 : index
      %35 = arith.remsi %34, %c4 : index
      %36 = arith.divsi %34, %c4 : index
      %37 = memref.load %arg3[%36] : memref<?xi32>
      %38 = arith.index_cast %37 : i32 to index
      %39 = memref.load %arg5[%38, %35] : memref<?x4xf32>
      memref.store %39, %arg6[%36, %35] : memref<?x4xf32>
      cf.br ^bb10
    ^bb10:  // 2 preds: ^bb8, ^bb9
      %40 = arith.addi %0, %c21 : index
      %41 = arith.remsi %40, %2 : index
      %c512_9 = arith.constant 512 : index
      %42 = arith.muli %41, %c512_9 : index
      %c-512_10 = arith.constant -512 : index
      %43 = arith.muli %41, %c-512_10 : index
      %44 = arith.addi %43, %dim_0 : index
      %c512_11 = arith.constant 512 : index
      %45 = arith.cmpi slt, %44, %c512_11 : index
      %46 = arith.select %45, %44, %c512_11 : index
      %47 = arith.cmpi slt, %1, %46 : index
      cf.cond_br %47, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      %48 = arith.addi %1, %42 : index
      memref.store %c0_i32, %arg7[%48] : memref<?xi32>
      cf.br ^bb12
    ^bb12:  // 2 preds: ^bb10, ^bb11
      %49 = arith.addi %0, %c28 : index
      %50 = arith.remsi %49, %2 : index
      %c512_12 = arith.constant 512 : index
      %51 = arith.muli %50, %c512_12 : index
      %c-512_13 = arith.constant -512 : index
      %52 = arith.muli %50, %c-512_13 : index
      %53 = arith.addi %52, %dim_0 : index
      %c512_14 = arith.constant 512 : index
      %54 = arith.cmpi slt, %53, %c512_14 : index
      %55 = arith.select %54, %53, %c512_14 : index
      %56 = arith.cmpi slt, %1, %55 : index
      cf.cond_br %56, ^bb13, ^bb17
    ^bb13:  // pred: ^bb12
      %57 = arith.addi %1, %51 : index
      cf.br ^bb14(%c0 : index)
    ^bb14(%58: index):  // 2 preds: ^bb13, ^bb15
      %59 = arith.cmpi slt, %58, %c2 : index
      cf.cond_br %59, ^bb15, ^bb16
    ^bb15:  // pred: ^bb14
      %60 = memref.load %arg1[%57, %58] : memref<?x2xi32>
      %61 = memref.load %arg7[%57] : memref<?xi32>
      %62 = arith.addi %61, %60 : i32
      memref.store %62, %arg7[%57] : memref<?xi32>
      %63 = arith.addi %58, %c1 : index
      cf.br ^bb14(%63 : index)
    ^bb16:  // pred: ^bb14
      cf.br ^bb17
    ^bb17:  // 2 preds: ^bb12, ^bb16
      %64 = arith.addi %0, %c35 : index
      %65 = arith.remsi %64, %2 : index
      %c512_15 = arith.constant 512 : index
      %66 = arith.muli %65, %c512_15 : index
      %c-512_16 = arith.constant -512 : index
      %67 = arith.muli %65, %c-512_16 : index
      %68 = arith.addi %67, %arg8 : index
      %c512_17 = arith.constant 512 : index
      %69 = arith.cmpi slt, %68, %c512_17 : index
      %70 = arith.select %69, %68, %c512_17 : index
      %71 = arith.cmpi slt, %1, %70 : index
      cf.cond_br %71, ^bb18, ^bb19
    ^bb18:  // pred: ^bb17
      %72 = arith.addi %1, %66 : index
      %73 = arith.remsi %72, %c4 : index
      %74 = arith.divsi %72, %c4 : index
      %75 = memref.load %arg7[%74] : memref<?xi32>
      %76 = arith.index_cast %75 : i32 to index
      %77 = memref.load %arg9[%76, %73] : memref<?x4xf32>
      memref.store %77, %arg10[%74, %73] : memref<?x4xf32>
      cf.br ^bb19
    ^bb19:  // 2 preds: ^bb17, ^bb18
      %78 = arith.addi %0, %c42 : index
      %79 = arith.remsi %78, %2 : index
      %c512_18 = arith.constant 512 : index
      %80 = arith.muli %79, %c512_18 : index
      %c-512_19 = arith.constant -512 : index
      %81 = arith.muli %79, %c-512_19 : index
      %82 = arith.addi %81, %dim_1 : index
      %c512_20 = arith.constant 512 : index
      %83 = arith.cmpi slt, %82, %c512_20 : index
      %84 = arith.select %83, %82, %c512_20 : index
      %85 = arith.cmpi slt, %1, %84 : index
      cf.cond_br %85, ^bb20, ^bb21
    ^bb20:  // pred: ^bb19
      %86 = arith.addi %1, %80 : index
      memref.store %c0_i32, %arg11[%86] : memref<?xi32>
      cf.br ^bb21
    ^bb21:  // 2 preds: ^bb19, ^bb20
      %87 = arith.addi %0, %c49 : index
      %88 = arith.remsi %87, %2 : index
      %c512_21 = arith.constant 512 : index
      %89 = arith.muli %88, %c512_21 : index
      %c-512_22 = arith.constant -512 : index
      %90 = arith.muli %88, %c-512_22 : index
      %91 = arith.addi %90, %dim_1 : index
      %c512_23 = arith.constant 512 : index
      %92 = arith.cmpi slt, %91, %c512_23 : index
      %93 = arith.select %92, %91, %c512_23 : index
      %94 = arith.cmpi slt, %1, %93 : index
      cf.cond_br %94, ^bb22, ^bb26
    ^bb22:  // pred: ^bb21
      %95 = arith.addi %1, %89 : index
      cf.br ^bb23(%c0 : index)
    ^bb23(%96: index):  // 2 preds: ^bb22, ^bb24
      %97 = arith.cmpi slt, %96, %c2 : index
      cf.cond_br %97, ^bb24, ^bb25
    ^bb24:  // pred: ^bb23
      %98 = memref.load %arg11[%95] : memref<?xi32>
      %99 = memref.load %arg2[%95, %96] : memref<?x2xi32>
      %100 = arith.addi %98, %99 : i32
      memref.store %100, %arg11[%95] : memref<?xi32>
      %101 = arith.addi %96, %c1 : index
      cf.br ^bb23(%101 : index)
    ^bb25:  // pred: ^bb23
      cf.br ^bb26
    ^bb26:  // 2 preds: ^bb21, ^bb25
      %102 = arith.addi %0, %c56 : index
      %103 = arith.remsi %102, %2 : index
      %c512_24 = arith.constant 512 : index
      %104 = arith.muli %103, %c512_24 : index
      %c-512_25 = arith.constant -512 : index
      %105 = arith.muli %103, %c-512_25 : index
      %106 = arith.addi %105, %arg12 : index
      %c512_26 = arith.constant 512 : index
      %107 = arith.cmpi slt, %106, %c512_26 : index
      %108 = arith.select %107, %106, %c512_26 : index
      %109 = arith.cmpi slt, %1, %108 : index
      cf.cond_br %109, ^bb27, ^bb28
    ^bb27:  // pred: ^bb26
      %110 = arith.addi %1, %104 : index
      %111 = arith.remsi %110, %c4 : index
      %112 = arith.divsi %110, %c4 : index
      %113 = memref.load %arg11[%112] : memref<?xi32>
      %114 = arith.index_cast %113 : i32 to index
      %115 = memref.load %arg13[%114, %111] : memref<?x4xf32>
      memref.store %115, %arg14[%112, %111] : memref<?x4xf32>
      cf.br ^bb28
    ^bb28:  // 2 preds: ^bb26, ^bb27
      gpu.return
    }
  }
}


