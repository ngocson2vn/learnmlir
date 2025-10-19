// -----// IR Dump After TileLoopsPass (tile-loops) ('func.func' operation: @add_two_vectors) //----- //
#map = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    scf.parallel (%arg3) = (%c0) to (%dim) step (%c128) {
      %0 = affine.min #map(%c128, %dim, %arg3)
      scf.parallel (%arg4) = (%c0) to (%0) step (%c1) {
        %1 = arith.addi %arg4, %arg3 : index
        %2 = memref.load %arg0[%1] : memref<?xf64>
        %3 = memref.load %arg1[%1] : memref<?xf64>
        %4 = arith.addf %2, %3 : f64
        memref.store %4, %arg2[%1] : memref<?xf64>
        scf.reduce 
      }
      scf.reduce 
    }
    return
  }
}


// -----// IR Dump After GpuMapParallelLoopsPass (gpu-map-parallel-loops) ('func.func' operation: @add_two_vectors) //----- //
#map = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    scf.parallel (%arg3) = (%c0) to (%dim) step (%c128) {
      %0 = affine.min #map(%c128, %dim, %arg3)
      scf.parallel (%arg4) = (%c0) to (%0) step (%c1) {
        %1 = arith.addi %arg4, %arg3 : index
        %2 = memref.load %arg0[%1] : memref<?xf64>
        %3 = memref.load %arg1[%1] : memref<?xf64>
        %4 = arith.addf %2, %3 : f64
        memref.store %4, %arg2[%1] : memref<?xf64>
        scf.reduce 
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
      scf.reduce 
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    return
  }
}


// -----// IR Dump After ConvertParallelLoopToGpuPass (convert-parallel-loops-to-gpu) ('builtin.module' operation) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (128, d1 - d2)>
module {
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %c1_0 = arith.constant 1 : index
    %0 = affine.apply #map(%dim)[%c0, %c128]
    %c128_1 = arith.constant 128 : index
    %1 = affine.apply #map(%c128_1)[%c0, %c1]
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %0, %arg10 = %c1_0, %arg11 = %c1_0) threads(%arg6, %arg7, %arg8) in (%arg12 = %1, %arg13 = %c1_0, %arg14 = %c1_0) {
      %2 = affine.apply #map1(%arg3)[%c128, %c0]
      %3 = affine.min #map2(%c128, %dim, %2)
      %4 = affine.apply #map1(%arg6)[%c1, %c0]
      %5 = arith.cmpi slt, %4, %3 : index
      scf.if %5 {
        %6 = arith.addi %4, %2 : index
        %7 = memref.load %arg0[%6] : memref<?xf64>
        %8 = memref.load %arg1[%6] : memref<?xf64>
        %9 = arith.addf %7, %8 : f64
        memref.store %9, %arg2[%6] : memref<?xf64>
      }
      gpu.terminator
    } {SCFToGPU_visited}
    return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 128)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0, s1] -> (s0 * -128 + s1, 128)>
module {
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %0 = affine.apply #map()[%dim]
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %0, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c128, %arg13 = %c1, %arg14 = %c1) {
      %1 = affine.apply #map1()[%arg3]
      %2 = affine.min #map2()[%arg3, %dim]
      %3 = arith.cmpi slt, %arg6, %2 : index
      scf.if %3 {
        %4 = arith.addi %arg6, %1 : index
        %5 = memref.load %arg0[%4] : memref<?xf64>
        %6 = memref.load %arg1[%4] : memref<?xf64>
        %7 = arith.addf %5, %6 : f64
        memref.store %7, %arg2[%4] : memref<?xf64>
      }
      gpu.terminator
    } {SCFToGPU_visited}
    return
  }
}


// -----// IR Dump After GpuLaunchSinkIndexComputationsPass (gpu-launch-sink-index-computations) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 128)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0, s1] -> (s0 * -128 + s1, 128)>
module {
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %0 = affine.apply #map()[%dim]
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %0, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c128, %arg13 = %c1, %arg14 = %c1) {
      %c0_0 = arith.constant 0 : index
      %dim_1 = memref.dim %arg0, %c0_0 : memref<?xf64>
      %1 = affine.apply #map1()[%arg3]
      %2 = affine.min #map2()[%arg3, %dim_1]
      %3 = arith.cmpi slt, %arg6, %2 : index
      scf.if %3 {
        %4 = arith.addi %arg6, %1 : index
        %5 = memref.load %arg0[%4] : memref<?xf64>
        %6 = memref.load %arg1[%4] : memref<?xf64>
        %7 = arith.addf %5, %6 : f64
        memref.store %7, %arg2[%4] : memref<?xf64>
      }
      gpu.terminator
    } {SCFToGPU_visited}
    return
  }
}


// -----// IR Dump After GpuKernelOutliningPass (gpu-kernel-outlining) ('builtin.module' operation) //----- //
#map = affine_map<()[s0] -> (s0 ceildiv 128)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0, s1] -> (s0 * -128 + s1, 128)>
module attributes {gpu.container_module} {
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %0 = affine.apply #map()[%dim]
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%0, %c1, %c1) threads in (%c128, %c1, %c1)  args(%arg0 : memref<?xf64>, %arg1 : memref<?xf64>, %arg2 : memref<?xf64>)
    return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %c0 = arith.constant 0 : index
      %dim = memref.dim %arg0, %c0 : memref<?xf64>
      %0 = affine.apply #map1()[%block_id_x]
      %1 = affine.min #map2()[%block_id_x, %dim]
      %2 = arith.cmpi slt, %thread_id_x, %1 : index
      scf.if %2 {
        %3 = arith.addi %thread_id_x, %0 : index
        %4 = memref.load %arg0[%3] : memref<?xf64>
        %5 = memref.load %arg1[%3] : memref<?xf64>
        %6 = arith.addf %4, %5 : f64
        memref.store %6, %arg2[%3] : memref<?xf64>
      }
      gpu.return
    }
  }
}


// -----// IR Dump After LowerAffinePass (lower-affine) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %c128_0 = arith.constant 128 : index
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %0 = arith.cmpi sle, %dim, %c0_1 : index
    %1 = arith.subi %c0_1, %dim : index
    %2 = arith.subi %dim, %c1_2 : index
    %3 = arith.select %0, %1, %2 : index
    %4 = arith.divsi %3, %c128_0 : index
    %5 = arith.subi %c0_1, %4 : index
    %6 = arith.addi %4, %c1_2 : index
    %7 = arith.select %0, %5, %6 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%7, %c1, %c1) threads in (%c128, %c1, %c1)  args(%arg0 : memref<?xf64>, %arg1 : memref<?xf64>, %arg2 : memref<?xf64>)
    return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %c0 = arith.constant 0 : index
      %dim = memref.dim %arg0, %c0 : memref<?xf64>
      %c128 = arith.constant 128 : index
      %0 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %c-128 = arith.constant -128 : index
      %1 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %2 = arith.addi %1, %dim : index
      %c128_0 = arith.constant 128 : index
      %3 = arith.minsi %2, %c128_0 : index
      %4 = arith.cmpi slt, %thread_id_x, %3 : index
      scf.if %4 {
        %5 = arith.addi %thread_id_x, %0 : index
        %6 = memref.load %arg0[%5] : memref<?xf64>
        %7 = memref.load %arg1[%5] : memref<?xf64>
        %8 = arith.addf %6, %7 : f64
        memref.store %8, %arg2[%5] : memref<?xf64>
      }
      gpu.return
    }
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %0 = arith.cmpi sle, %dim, %c0 : index
    %1 = arith.subi %c0, %dim : index
    %2 = arith.subi %dim, %c1 : index
    %3 = arith.select %0, %1, %2 : index
    %4 = arith.divsi %3, %c128 : index
    %5 = arith.subi %c0, %4 : index
    %6 = arith.addi %4, %c1 : index
    %7 = arith.select %0, %5, %6 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%7, %c1, %c1) threads in (%c128, %c1, %c1)  args(%arg0 : memref<?xf64>, %arg1 : memref<?xf64>, %arg2 : memref<?xf64>)
    return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %c-128 = arith.constant -128 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %dim = memref.dim %arg0, %c0 : memref<?xf64>
      %0 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %1 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %2 = arith.addi %1, %dim : index
      %3 = arith.minsi %2, %c128 : index
      %4 = arith.cmpi slt, %thread_id_x, %3 : index
      scf.if %4 {
        %5 = arith.addi %thread_id_x, %0 : index
        %6 = memref.load %arg0[%5] : memref<?xf64>
        %7 = memref.load %arg1[%5] : memref<?xf64>
        %8 = arith.addf %6, %7 : f64
        memref.store %8, %arg2[%5] : memref<?xf64>
      }
      gpu.return
    }
  }
}


// -----// IR Dump After SCFToControlFlowPass (convert-scf-to-cf) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %0 = arith.cmpi sle, %dim, %c0 : index
    %1 = arith.subi %c0, %dim : index
    %2 = arith.subi %dim, %c1 : index
    %3 = arith.select %0, %1, %2 : index
    %4 = arith.divsi %3, %c128 : index
    %5 = arith.subi %c0, %4 : index
    %6 = arith.addi %4, %c1 : index
    %7 = arith.select %0, %5, %6 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%7, %c1, %c1) threads in (%c128, %c1, %c1)  args(%arg0 : memref<?xf64>, %arg1 : memref<?xf64>, %arg2 : memref<?xf64>)
    return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    gpu.func @add_two_vectors_kernel(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %c-128 = arith.constant -128 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %dim = memref.dim %arg0, %c0 : memref<?xf64>
      %0 = arith.muli %block_id_x, %c128 overflow<nsw> : index
      %1 = arith.muli %block_id_x, %c-128 overflow<nsw> : index
      %2 = arith.addi %1, %dim : index
      %3 = arith.minsi %2, %c128 : index
      %4 = arith.cmpi slt, %thread_id_x, %3 : index
      cf.cond_br %4, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %5 = arith.addi %thread_id_x, %0 : index
      %6 = memref.load %arg0[%5] : memref<?xf64>
      %7 = memref.load %arg1[%5] : memref<?xf64>
      %8 = arith.addf %6, %7 : f64
      memref.store %8, %arg2[%5] : memref<?xf64>
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      gpu.return
    }
  }
}


// -----// IR Dump After ConvertGpuOpsToNVVMOps (convert-gpu-to-nvvm) ('gpu.module' operation: @add_two_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %0 = arith.cmpi sle, %dim, %c0 : index
    %1 = arith.subi %c0, %dim : index
    %2 = arith.subi %dim, %c1 : index
    %3 = arith.select %0, %1, %2 : index
    %4 = arith.divsi %3, %c128 : index
    %5 = arith.subi %c0, %4 : index
    %6 = arith.addi %4, %c1 : index
    %7 = arith.select %0, %5, %6 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%7, %c1, %c1) threads in (%c128, %c1, %c1)  args(%arg0 : memref<?xf64>, %arg1 : memref<?xf64>, %arg2 : memref<?xf64>)
    return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg10, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %2 = llvm.insertvalue %arg11, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %3 = llvm.insertvalue %arg12, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %4 = llvm.insertvalue %arg13, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %5 = llvm.insertvalue %arg14, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %6 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %12 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %13 = llvm.insertvalue %arg0, %12[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %14 = llvm.insertvalue %arg1, %13[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %15 = llvm.insertvalue %arg2, %14[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %16 = llvm.insertvalue %arg3, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %17 = llvm.insertvalue %arg4, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %18 = llvm.mlir.constant(-128 : index) : i64
      %19 = llvm.mlir.constant(128 : index) : i64
      %20 = llvm.mlir.constant(0 : index) : i64
      %21 = nvvm.read.ptx.sreg.ctaid.x : i32
      %22 = llvm.sext %21 : i32 to i64
      %23 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %24 = llvm.sext %23 : i32 to i64
      %25 = llvm.extractvalue %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %26 = llvm.mul %22, %19 overflow<nsw> : i64
      %27 = llvm.mul %22, %18 overflow<nsw> : i64
      %28 = llvm.add %27, %25 : i64
      %29 = llvm.intr.smin(%28, %19) : (i64, i64) -> i64
      %30 = llvm.icmp "slt" %24, %29 : i64
      llvm.cond_br %30, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %31 = llvm.add %24, %26 : i64
      %32 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %33 = llvm.getelementptr inbounds|nuw %32[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %34 = llvm.load %33 : !llvm.ptr -> f64
      %35 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %36 = llvm.getelementptr inbounds|nuw %35[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %37 = llvm.load %36 : !llvm.ptr -> f64
      %38 = llvm.fadd %34, %37 : f64
      %39 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %40 = llvm.getelementptr inbounds|nuw %39[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %38, %40 : f64, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
}


// -----// IR Dump After NVVMOptimizeForTargetPass (llvm-optimize-for-nvvm-target) ('gpu.module' operation: @add_two_vectors_kernel) //----- //
module attributes {gpu.container_module} {
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %0 = arith.cmpi sle, %dim, %c0 : index
    %1 = arith.subi %c0, %dim : index
    %2 = arith.subi %dim, %c1 : index
    %3 = arith.select %0, %1, %2 : index
    %4 = arith.divsi %3, %c128 : index
    %5 = arith.subi %c0, %4 : index
    %6 = arith.addi %4, %c1 : index
    %7 = arith.select %0, %5, %6 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%7, %c1, %c1) threads in (%c128, %c1, %c1)  args(%arg0 : memref<?xf64>, %arg1 : memref<?xf64>, %arg2 : memref<?xf64>)
    return
  }
  gpu.module @add_two_vectors_kernel attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i64>} {
    llvm.func @add_two_vectors_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64) attributes {gpu.kernel, gpu.known_block_size = array<i32: 128, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 128, 1, 1>} {
      %0 = llvm.mlir.constant(128 : index) : i64
      %1 = llvm.mlir.constant(-128 : index) : i64
      %2 = nvvm.read.ptx.sreg.ctaid.x : i32
      %3 = llvm.sext %2 : i32 to i64
      %4 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 128> : i32
      %5 = llvm.sext %4 : i32 to i64
      %6 = llvm.mul %3, %0 overflow<nsw> : i64
      %7 = llvm.mul %3, %1 overflow<nsw> : i64
      %8 = llvm.add %7, %arg3 : i64
      %9 = llvm.intr.smin(%8, %0) : (i64, i64) -> i64
      %10 = llvm.icmp "slt" %5, %9 : i64
      llvm.cond_br %10, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %11 = llvm.add %5, %6 : i64
      %12 = llvm.getelementptr inbounds|nuw %arg1[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %13 = llvm.load %12 : !llvm.ptr -> f64
      %14 = llvm.getelementptr inbounds|nuw %arg6[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %15 = llvm.load %14 : !llvm.ptr -> f64
      %16 = llvm.fadd %13, %15 : f64
      %17 = llvm.getelementptr inbounds|nuw %arg11[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %16, %17 : f64, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
}


// -----// IR Dump After GpuModuleToCubinPass (gpu-to-cubin) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  gpu.binary @add_two_vectors_kernel <#gpu.select_object<#nvvm.target<O = 3, chip = "sm_86", features = "+ptx84">>> [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", features = "+ptx84">, bin = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00|\00\00\00\00\00\00\00\00\00\00\00\00\0D\00\00\00\00\00\00\00\0A\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00add_two_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BA\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DF\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00+\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00:\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\04\04\00\00\00\04 \00\00\00\0C\81\80\80(\00\048\00\00\00\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0C\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\047\04\00|\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01x\00\03\19x\00\04\17\0C\00\00\00\00\00\0E\00p\00\00\F0!\00\04\17\0C\00\00\00\00\00\0D\00h\00\00\F0!\00\04\17\0C\00\00\00\00\00\0C\00`\00\00\F0!\00\04\17\0C\00\00\00\00\00\0B\00X\00\00\F5!\00\04\17\0C\00\00\00\00\00\0A\00P\00\00\F5!\00\04\17\0C\00\00\00\00\00\09\00H\00\00\F0!\00\04\17\0C\00\00\00\00\00\08\00@\00\00\F0!\00\04\17\0C\00\00\00\00\00\07\008\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F5!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F5!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F5!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F5!\00\03\1B\FF\00\04\1C\08\00\80\00\00\00p\01\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056D\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\E4\0F\00\19y\02\00\00\00\00\00\00%\00\00\00(\0E\00\19y\05\00\00\00\00\00\00!\00\00\00b\0E\00%x\02\02\80\00\00\00\FF\00\8E\07\00\CA\1F\00\10z\00\02\00^\00\00\FF\E1\F3\07\00\C8\0F\00\0Cr\00\00\05\00\00\00p@\F0\03\00\E4/\00\10z\00\03\00_\00\00\FF\E5\FF\00\00\C8\0F\00\0Cr\00\00\FF\00\00\00\00C\F0\03\00\DA\0F\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\12r\05\02\05\00\00\00\FF\FC\8E\07\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19x\00\05\03\00\00\00\03\02\01\00\00\E2\0F\04$x\08\05\08\00\00\00\FF\00\8E\07\00\CA\0F\00\10z\04\08\00d\00\00\FF\E0\F3\07\00\E4\0F\04\10z\06\08\00Z\00\00\FF\E0\F1\07\00\E4\0F\00\10z\05\00\00e\00\00\FF\E4\FF\00\00\E4\0F\04\10z\07\00\00[\00\00\FF\E4\7F\00\00\C8\0F\00\81y\04\04\04\00\00\00\00\1B\1E\0C\00\A8\0E\00\81y\02\06\04\00\00\00\00\1B\1E\0C\00\A2\0E\00\10z\08\08\00n\00\00\FF\E0\F1\07\00\C8\0F\00\10z\09\00\00o\00\00\FF\E4\7F\00\00\E2\0F\00)r\02\02\00\00\00\00\04\00\00\00\00\0EN\00\86y\00\08\02\00\00\00\04\1B\10\0C\00\E2\1F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00Q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00,\01\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\05\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00h\05\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\05\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\88\05\00\00\00\00\00\00\D8\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\07\00\00\00\00\00\00\80\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0C\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\00\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\88\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\04\00\00\00\00\00\00x\04\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\00\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
  func.func @add_two_vectors(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    %0 = arith.cmpi sle, %dim, %c0 : index
    %1 = arith.subi %c0, %dim : index
    %2 = arith.subi %dim, %c1 : index
    %3 = arith.select %0, %1, %2 : index
    %4 = arith.divsi %3, %c128 : index
    %5 = arith.subi %c0, %4 : index
    %6 = arith.addi %4, %c1 : index
    %7 = arith.select %0, %5, %6 : index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%7, %c1, %c1) threads in (%c128, %c1, %c1)  args(%arg0 : memref<?xf64>, %arg1 : memref<?xf64>, %arg2 : memref<?xf64>)
    return
  }
}


// -----// IR Dump After ConvertToLLVMPass (convert-to-llvm) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  gpu.binary @add_two_vectors_kernel <#gpu.select_object<#nvvm.target<O = 3, chip = "sm_86", features = "+ptx84">>> [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", features = "+ptx84">, bin = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00|\00\00\00\00\00\00\00\00\00\00\00\00\0D\00\00\00\00\00\00\00\0A\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00add_two_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BA\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DF\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00+\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00:\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\04\04\00\00\00\04 \00\00\00\0C\81\80\80(\00\048\00\00\00\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0C\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\047\04\00|\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01x\00\03\19x\00\04\17\0C\00\00\00\00\00\0E\00p\00\00\F0!\00\04\17\0C\00\00\00\00\00\0D\00h\00\00\F0!\00\04\17\0C\00\00\00\00\00\0C\00`\00\00\F0!\00\04\17\0C\00\00\00\00\00\0B\00X\00\00\F5!\00\04\17\0C\00\00\00\00\00\0A\00P\00\00\F5!\00\04\17\0C\00\00\00\00\00\09\00H\00\00\F0!\00\04\17\0C\00\00\00\00\00\08\00@\00\00\F0!\00\04\17\0C\00\00\00\00\00\07\008\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F5!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F5!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F5!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F5!\00\03\1B\FF\00\04\1C\08\00\80\00\00\00p\01\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056D\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\E4\0F\00\19y\02\00\00\00\00\00\00%\00\00\00(\0E\00\19y\05\00\00\00\00\00\00!\00\00\00b\0E\00%x\02\02\80\00\00\00\FF\00\8E\07\00\CA\1F\00\10z\00\02\00^\00\00\FF\E1\F3\07\00\C8\0F\00\0Cr\00\00\05\00\00\00p@\F0\03\00\E4/\00\10z\00\03\00_\00\00\FF\E5\FF\00\00\C8\0F\00\0Cr\00\00\FF\00\00\00\00C\F0\03\00\DA\0F\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\12r\05\02\05\00\00\00\FF\FC\8E\07\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19x\00\05\03\00\00\00\03\02\01\00\00\E2\0F\04$x\08\05\08\00\00\00\FF\00\8E\07\00\CA\0F\00\10z\04\08\00d\00\00\FF\E0\F3\07\00\E4\0F\04\10z\06\08\00Z\00\00\FF\E0\F1\07\00\E4\0F\00\10z\05\00\00e\00\00\FF\E4\FF\00\00\E4\0F\04\10z\07\00\00[\00\00\FF\E4\7F\00\00\C8\0F\00\81y\04\04\04\00\00\00\00\1B\1E\0C\00\A8\0E\00\81y\02\06\04\00\00\00\00\1B\1E\0C\00\A2\0E\00\10z\08\08\00n\00\00\FF\E0\F1\07\00\C8\0F\00\10z\09\00\00o\00\00\FF\E4\7F\00\00\E2\0F\00)r\02\02\00\00\00\00\04\00\00\00\00\0EN\00\86y\00\08\02\00\00\00\04\1B\10\0C\00\E2\1F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00Q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00,\01\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\05\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00h\05\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\05\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\88\05\00\00\00\00\00\00\D8\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\07\00\00\00\00\00\00\80\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0C\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\00\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\88\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\04\00\00\00\00\00\00x\04\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\00\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64) {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg10, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg11, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg12, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg13, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg14, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = builtin.unrealized_conversion_cast %5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %7 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg5, %7[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg6, %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg7, %9[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg8, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg9, %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = builtin.unrealized_conversion_cast %12 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %14 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %15 = llvm.insertvalue %arg0, %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %arg1, %15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %arg2, %16[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg3, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg4, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = builtin.unrealized_conversion_cast %19 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %21 = llvm.mlir.constant(128 : index) : i64
    %22 = builtin.unrealized_conversion_cast %21 : i64 to index
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = builtin.unrealized_conversion_cast %23 : i64 to index
    %25 = llvm.mlir.constant(0 : index) : i64
    %26 = llvm.extractvalue %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.icmp "sle" %26, %25 : i64
    %28 = llvm.sub %25, %26 : i64
    %29 = llvm.sub %26, %23 : i64
    %30 = llvm.select %27, %28, %29 : i1, i64
    %31 = llvm.sdiv %30, %21 : i64
    %32 = llvm.sub %25, %31 : i64
    %33 = llvm.add %31, %23 : i64
    %34 = llvm.select %27, %32, %33 : i1, i64
    %35 = builtin.unrealized_conversion_cast %34 : i64 to index
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%35, %24, %24) threads in (%22, %24, %24)  args(%20 : memref<?xf64>, %13 : memref<?xf64>, %6 : memref<?xf64>)
    llvm.return
  }
}


// -----// IR Dump After GpuToLLVMConversionPass (gpu-to-llvm) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  gpu.binary @add_two_vectors_kernel <#gpu.select_object<#nvvm.target<O = 3, chip = "sm_86", features = "+ptx84">>> [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", features = "+ptx84">, bin = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00|\00\00\00\00\00\00\00\00\00\00\00\00\0D\00\00\00\00\00\00\00\0A\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00add_two_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BA\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DF\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00+\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00:\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\04\04\00\00\00\04 \00\00\00\0C\81\80\80(\00\048\00\00\00\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0C\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\047\04\00|\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01x\00\03\19x\00\04\17\0C\00\00\00\00\00\0E\00p\00\00\F0!\00\04\17\0C\00\00\00\00\00\0D\00h\00\00\F0!\00\04\17\0C\00\00\00\00\00\0C\00`\00\00\F0!\00\04\17\0C\00\00\00\00\00\0B\00X\00\00\F5!\00\04\17\0C\00\00\00\00\00\0A\00P\00\00\F5!\00\04\17\0C\00\00\00\00\00\09\00H\00\00\F0!\00\04\17\0C\00\00\00\00\00\08\00@\00\00\F0!\00\04\17\0C\00\00\00\00\00\07\008\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F5!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F5!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F5!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F5!\00\03\1B\FF\00\04\1C\08\00\80\00\00\00p\01\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056D\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\E4\0F\00\19y\02\00\00\00\00\00\00%\00\00\00(\0E\00\19y\05\00\00\00\00\00\00!\00\00\00b\0E\00%x\02\02\80\00\00\00\FF\00\8E\07\00\CA\1F\00\10z\00\02\00^\00\00\FF\E1\F3\07\00\C8\0F\00\0Cr\00\00\05\00\00\00p@\F0\03\00\E4/\00\10z\00\03\00_\00\00\FF\E5\FF\00\00\C8\0F\00\0Cr\00\00\FF\00\00\00\00C\F0\03\00\DA\0F\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\12r\05\02\05\00\00\00\FF\FC\8E\07\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19x\00\05\03\00\00\00\03\02\01\00\00\E2\0F\04$x\08\05\08\00\00\00\FF\00\8E\07\00\CA\0F\00\10z\04\08\00d\00\00\FF\E0\F3\07\00\E4\0F\04\10z\06\08\00Z\00\00\FF\E0\F1\07\00\E4\0F\00\10z\05\00\00e\00\00\FF\E4\FF\00\00\E4\0F\04\10z\07\00\00[\00\00\FF\E4\7F\00\00\C8\0F\00\81y\04\04\04\00\00\00\00\1B\1E\0C\00\A8\0E\00\81y\02\06\04\00\00\00\00\1B\1E\0C\00\A2\0E\00\10z\08\08\00n\00\00\FF\E0\F1\07\00\C8\0F\00\10z\09\00\00o\00\00\FF\E4\7F\00\00\E2\0F\00)r\02\02\00\00\00\00\04\00\00\00\00\0EN\00\86y\00\08\02\00\00\00\04\1B\10\0C\00\E2\1F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00Q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00,\01\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\05\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00h\05\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\05\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\88\05\00\00\00\00\00\00\D8\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\07\00\00\00\00\00\00\80\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0C\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\00\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\88\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\04\00\00\00\00\00\00x\04\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\00\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64) {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(128 : index) : i64
    %3 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %4 = llvm.insertvalue %arg10, %3[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg11, %4[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.insertvalue %arg12, %5[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %7 = llvm.insertvalue %arg13, %6[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg14, %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = builtin.unrealized_conversion_cast %8 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %10 = llvm.insertvalue %arg5, %3[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg6, %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg7, %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.insertvalue %arg8, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %arg9, %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = builtin.unrealized_conversion_cast %14 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %16 = llvm.insertvalue %arg0, %3[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %arg1, %16[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg2, %17[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.insertvalue %arg3, %18[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %arg4, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = builtin.unrealized_conversion_cast %20 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %22 = builtin.unrealized_conversion_cast %2 : i64 to index
    %23 = builtin.unrealized_conversion_cast %1 : i64 to index
    %24 = llvm.icmp "sle" %arg3, %0 : i64
    %25 = llvm.sub %0, %arg3 : i64
    %26 = llvm.sub %arg3, %1 : i64
    %27 = llvm.select %24, %25, %26 : i1, i64
    %28 = llvm.sdiv %27, %2 : i64
    %29 = llvm.sub %0, %28 : i64
    %30 = llvm.add %28, %1 : i64
    %31 = llvm.select %24, %29, %30 : i1, i64
    %32 = builtin.unrealized_conversion_cast %31 : i64 to index
    %33 = llvm.extractvalue %20[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.extractvalue %20[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.extractvalue %20[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.extractvalue %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.extractvalue %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %38 = llvm.extractvalue %14[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.extractvalue %14[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.extractvalue %14[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %42 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %46 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%31, %1, %1) threads in (%2, %1, %1) : i64 args(%33 : !llvm.ptr, %34 : !llvm.ptr, %35 : i64, %36 : i64, %37 : i64, %38 : !llvm.ptr, %39 : !llvm.ptr, %40 : i64, %41 : i64, %42 : i64, %43 : !llvm.ptr, %44 : !llvm.ptr, %45 : i64, %46 : i64, %47 : i64)
    llvm.return
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module attributes {gpu.container_module} {
  gpu.binary @add_two_vectors_kernel <#gpu.select_object<#nvvm.target<O = 3, chip = "sm_86", features = "+ptx84">>> [#gpu.object<#nvvm.target<O = 3, chip = "sm_86", features = "+ptx84">, bin = "\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00|\00\00\00\00\00\00\00\00\00\00\00\00\0D\00\00\00\00\00\00\00\0A\00\00\00\00\00\00V\05V\00@\008\00\03\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.add_two_vectors_kernel\00.nv.info.add_two_vectors_kernel\00.nv.shared.add_two_vectors_kernel\00.rel.nv.constant0.add_two_vectors_kernel\00.nv.constant0.add_two_vectors_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00add_two_vectors_kernel\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\BA\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\DF\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00+\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00:\01\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\FF\FF\FF\FF$\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\FF\FF\FF\FF4\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\02\00\00\00\00\00\00\04\04\00\00\00\04 \00\00\00\0C\81\80\80(\00\048\00\00\00\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0C\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\047\04\00|\00\00\00\015\00\00\04\0A\08\00\02\00\00\00`\01x\00\03\19x\00\04\17\0C\00\00\00\00\00\0E\00p\00\00\F0!\00\04\17\0C\00\00\00\00\00\0D\00h\00\00\F0!\00\04\17\0C\00\00\00\00\00\0C\00`\00\00\F0!\00\04\17\0C\00\00\00\00\00\0B\00X\00\00\F5!\00\04\17\0C\00\00\00\00\00\0A\00P\00\00\F5!\00\04\17\0C\00\00\00\00\00\09\00H\00\00\F0!\00\04\17\0C\00\00\00\00\00\08\00@\00\00\F0!\00\04\17\0C\00\00\00\00\00\07\008\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F5!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F5!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F5!\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F5!\00\03\1B\FF\00\04\1C\08\00\80\00\00\00p\01\00\00\04\05\0C\00\80\00\00\00\01\00\00\00\01\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00\FC\FF\FF\FF\00\00\00\00s\00\00\00\00\00\00\00\00\00\00\11%\00\056D\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00$v\01\FF\00\0A\00\00\FF\00\8E\07\00\E4\0F\00\19y\02\00\00\00\00\00\00%\00\00\00(\0E\00\19y\05\00\00\00\00\00\00!\00\00\00b\0E\00%x\02\02\80\00\00\00\FF\00\8E\07\00\CA\1F\00\10z\00\02\00^\00\00\FF\E1\F3\07\00\C8\0F\00\0Cr\00\00\05\00\00\00p@\F0\03\00\E4/\00\10z\00\03\00_\00\00\FF\E5\FF\00\00\C8\0F\00\0Cr\00\00\FF\00\00\00\00C\F0\03\00\DA\0F\00M\89\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00\12r\05\02\05\00\00\00\FF\FC\8E\07\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\C6\0F\00\19x\00\05\03\00\00\00\03\02\01\00\00\E2\0F\04$x\08\05\08\00\00\00\FF\00\8E\07\00\CA\0F\00\10z\04\08\00d\00\00\FF\E0\F3\07\00\E4\0F\04\10z\06\08\00Z\00\00\FF\E0\F1\07\00\E4\0F\00\10z\05\00\00e\00\00\FF\E4\FF\00\00\E4\0F\04\10z\07\00\00[\00\00\FF\E4\7F\00\00\C8\0F\00\81y\04\04\04\00\00\00\00\1B\1E\0C\00\A8\0E\00\81y\02\06\04\00\00\00\00\1B\1E\0C\00\A2\0E\00\10z\08\08\00n\00\00\FF\E0\F1\07\00\C8\0F\00\10z\09\00\00o\00\00\FF\E4\7F\00\00\E2\0F\00)r\02\02\00\00\00\00\04\00\00\00\00\0EN\00\86y\00\08\02\00\00\00\04\1B\10\0C\00\E2\1F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00Q\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D0\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\E8\03\00\00\00\00\00\000\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\04\00\00\00\00\00\00,\01\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00D\05\00\00\00\00\00\00 \00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00h\05\00\00\00\00\00\00\10\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00@\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\05\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\88\05\00\00\00\00\00\00\D8\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\07\00\00\00\00\00\00\80\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0C\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\06\00\00\00\05\00\00\00\00\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\88\05\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00x\04\00\00\00\00\00\00x\04\00\00\00\00\00\00\08\00\00\00\00\00\00\00\01\00\00\00\05\00\00\00\00\0D\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00">]
  llvm.func @add_two_vectors(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64) {
    %0 = llvm.mlir.constant(0 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(128 : index) : i64
    %3 = llvm.icmp "sle" %arg3, %0 : i64
    %4 = llvm.sub %0, %arg3 : i64
    %5 = llvm.sub %arg3, %1 : i64
    %6 = llvm.select %3, %4, %5 : i1, i64
    %7 = llvm.sdiv %6, %2 : i64
    %8 = llvm.sub %0, %7 : i64
    %9 = llvm.add %7, %1 : i64
    %10 = llvm.select %3, %8, %9 : i1, i64
    gpu.launch_func  @add_two_vectors_kernel::@add_two_vectors_kernel blocks in (%10, %1, %1) threads in (%2, %1, %1) : i64 args(%arg0 : !llvm.ptr, %arg1 : !llvm.ptr, %arg2 : i64, %arg3 : i64, %arg4 : i64, %arg5 : !llvm.ptr, %arg6 : !llvm.ptr, %arg7 : i64, %arg8 : i64, %arg9 : i64, %arg10 : !llvm.ptr, %arg11 : !llvm.ptr, %arg12 : i64, %arg13 : i64, %arg14 : i64)
    llvm.return
  }
}


