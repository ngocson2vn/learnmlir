// -----// IR Dump After DiscLhloLegalizeRootsToParallelLoopsPass (disc-lhlo-legalize-roots-to-parallel-loops) //----- //
func.func @shape_constraint_graph() {
  %c4 = arith.constant 4 : index
  %0 = "disc_shape.dim"() {name = @S2} : () -> index
  %1 = "disc_shape.dim"() {name = @S3} : () -> index
  "disc_shape.tie_product_equal"(%c4, %0, %1) {operand_segment_sizes = array<i32: 2, 1>} : (index, index, index) -> ()
  return
}

// -----// IR Dump After DiscLhloLegalizeRootsToParallelLoopsPass (disc-lhlo-legalize-roots-to-parallel-loops) //----- //
func.func @main(%arg0: !disc_ral.context) attributes {tf.entry_function = {input_placements = "cpu,cpu", inputs = "arg0,arg1", output_placements = "cpu", outputs = "out0"}} {
  %0 = llvm.mlir.constant(0 : i32) : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %1 = "disc_ral.dispatch"(%arg0, %c0) {backend_config = "", call_target_name = "ral_recv_input", device = "cpu", has_side_effect = false} : (!disc_ral.context, index) -> memref<?x4xf32>
  %2 = "disc_ral.dispatch"(%arg0, %c1) {backend_config = "", call_target_name = "ral_recv_input", device = "cpu", has_side_effect = false} : (!disc_ral.context, index) -> memref<?x4xf32>
  %dim = memref.dim %2, %c0 : memref<?x4xf32>
  %dim_0 = memref.dim %1, %c0 : memref<?x4xf32>
  %reinterpret_cast = memref.reinterpret_cast %1 to offset: [0], sizes: [%dim_0, 4], strides: [4, 1] {kDiscSymbolicDimAttr = [@S0, @C4]} : memref<?x4xf32> to memref<?x4xf32>
  %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [0], sizes: [%dim, 4], strides: [4, 1] {kDiscSymbolicDimAttr = [@S1, @C4]} : memref<?x4xf32> to memref<?x4xf32>
  %3 = arith.cmpi eq, %dim, %c1 : index
  %4 = arith.select %3, %dim_0, %dim : index
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<2xindex>
  memref.store %4, %alloca[%c0] : memref<2xindex>
  memref.store %c4, %alloca[%c1] : memref<2xindex>
  %alloc = memref.alloc(%dim_0) {kDiscSymbolicDimAttr = [@S0, @C4]} : memref<?x4xf32, #gpu.address_space<global>>
  %5 = llvm.inttoptr %0 : i32 to !llvm.ptr<i8>
  "disc_ral.dispatch"(%arg0, %5, %reinterpret_cast, %alloc) {backend_config = "", call_target_name = "h2d", device = "gpu", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>, memref<?x4xf32>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
  "disc_ral.dispatch"(%arg0, %5) {backend_config = "", call_target_name = "sync_on_stream", device = "gpu", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>) -> ()
  %alloc_2 = memref.alloc(%dim) {kDiscSymbolicDimAttr = [@S1, @C4]} : memref<?x4xf32, #gpu.address_space<global>>
  %6 = llvm.inttoptr %0 : i32 to !llvm.ptr<i8>
  "disc_ral.dispatch"(%arg0, %6, %reinterpret_cast_1, %alloc_2) {backend_config = "", call_target_name = "h2d", device = "gpu", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>, memref<?x4xf32>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
  "disc_ral.dispatch"(%arg0, %6) {backend_config = "", call_target_name = "sync_on_stream", device = "gpu", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>) -> ()
  %7 = arith.index_cast %4 : index to i32
  %8 = arith.muli %7, %c4_i32 : i32
  %alloca_3 = memref.alloca() {alignment = 64 : i64} : memref<2xi32>
  memref.store %8, %alloca_3[%c0] : memref<2xi32>
  memref.store %c1_i32, %alloca_3[%c1] : memref<2xi32>
  %9 = arith.index_cast %8 : i32 to index
  %alloc_4 = memref.alloc() : memref<f32, #gpu.address_space<global>>
  %alloc_5 = memref.alloc(%4) {kDiscSymbolicDimAttr = [@S2, @C4]} : memref<?x4xf32, #gpu.address_space<global>>
  %alloc_6 = memref.alloc(%4) {kDiscSymbolicDimAttr = [@S2, @C4]} : memref<?x4xf32, #gpu.address_space<global>>
  %alloc_7 = memref.alloc(%4) {kDiscSymbolicDimAttr = [@S2, @C4]} : memref<?x4xf32, #gpu.address_space<global>>
  %alloc_8 = memref.alloc(%9) {kDiscSymbolicDimAttr = [@S3, @C1]} : memref<?x1xf32, #gpu.address_space<global>>
  %alloc_9 = memref.alloc() : memref<1xf32, #gpu.address_space<global>>
  %c0_10 = arith.constant 0 : index
  %dim_11 = memref.dim %alloca_3, %c0_10 : memref<2xi32>
  "lmhlo_disc.printf"(%dim_11) {format = "main_kColReduction_reduce__6_1_0 operand 0 dims: %d \0A"} : (index) -> ()
  %c0_12 = arith.constant 0 : index
  %dim_13 = memref.dim %alloca, %c0_12 : memref<2xindex>
  "lmhlo_disc.printf"(%dim_13) {format = "main_kColReduction_reduce__6_1_0 operand 1 dims: %d \0A"} : (index) -> ()
  %c0_14 = arith.constant 0 : index
  %dim_15 = memref.dim %alloc, %c0_14 : memref<?x4xf32, #gpu.address_space<global>>
  %c1_16 = arith.constant 1 : index
  %dim_17 = memref.dim %alloc, %c1_16 : memref<?x4xf32, #gpu.address_space<global>>
  "lmhlo_disc.printf"(%dim_15, %dim_17) {format = "main_kColReduction_reduce__6_1_0 operand 2 dims: %d %d \0A"} : (index, index) -> ()
  %c0_18 = arith.constant 0 : index
  %dim_19 = memref.dim %alloc_2, %c0_18 : memref<?x4xf32, #gpu.address_space<global>>
  %c1_20 = arith.constant 1 : index
  %dim_21 = memref.dim %alloc_2, %c1_20 : memref<?x4xf32, #gpu.address_space<global>>
  "lmhlo_disc.printf"(%dim_19, %dim_21) {format = "main_kColReduction_reduce__6_1_0 operand 3 dims: %d %d \0A"} : (index, index) -> ()
  %c0_22 = arith.constant 0 : index
  %dim_23 = memref.dim %alloc_9, %c0_22 : memref<1xf32, #gpu.address_space<global>>
  "lmhlo_disc.printf"(%dim_23) {format = "main_kColReduction_reduce__6_1_0 result 0 dims: %d \0A"} : (index) -> ()
  "lmhlo.fusion"() ({
    "lmhlo.constant"(%alloc_4) {disc.device = "gpu", value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, #gpu.address_space<global>>) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%alloc, %alloca, %alloc_5) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>, disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<2xindex>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
    "lmhlo.dynamic_broadcast_in_dim"(%alloc_2, %alloca, %alloc_6) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>, disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<2xindex>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
    "lmhlo.multiply"(%alloc_5, %alloc_6, %alloc_7) {disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<?x4xf32, #gpu.address_space<global>>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
    "lmhlo.dynamic_reshape"(%alloc_7, %alloca_3, %alloc_8) {disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<2xi32>, memref<?x1xf32, #gpu.address_space<global>>) -> ()
    %c0_27 = arith.constant 0 : index
    %c1_28 = arith.constant 1 : index
    %c1_29 = arith.constant 1 : index
    %c0_30 = arith.constant 0 : index
    %dim_31 = memref.dim %alloc_9, %c0_30 : memref<1xf32, #gpu.address_space<global>>
    %13 = arith.muli %c1_29, %dim_31 : index
    scf.parallel (%arg1) = (%c0_27) to (%13) step (%c1_28) {
      %c1_36 = arith.constant 1 : index
      %17 = "disc_shape.delinearize"(%arg1, %c1_36) : (index, index) -> index
      %18 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
      memref.store %18, %alloc_9[%17] : memref<1xf32, #gpu.address_space<global>>
      scf.yield
    }
    %c0_32 = arith.constant 0 : index
    %c1_33 = arith.constant 1 : index
    %dim_34 = memref.dim %alloc_8, %c0_32 : memref<?x1xf32, #gpu.address_space<global>>
    %dim_35 = memref.dim %alloc_8, %c1_33 : memref<?x1xf32, #gpu.address_space<global>>
    %c512 = arith.constant 512 : index
    %c32 = arith.constant 32 : index
    %14 = arith.ceildivui %dim_35, %c512 : index
    %15 = arith.ceildivui %dim_34, %c32 : index
    %16 = arith.muli %14, %15 : index
    scf.parallel (%arg1, %arg2) = (%c0_32, %c0_32) to (%16, %c512) step (%c1_33, %c1_33) {
      %17 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
      %18 = arith.divui %arg1, %14 : index
      %19 = arith.remui %arg1, %14 : index
      %20 = arith.muli %19, %c512 : index
      %21 = arith.addi %20, %arg2 : index
      %22 = arith.cmpi ult, %21, %dim_35 : index
      %23 = scf.if %22 -> (f32) {
        %24 = scf.for %arg3 = %c0_32 to %c32 step %c1_33 iter_args(%arg4 = %17) -> (f32) {
          %25 = arith.muli %18, %c32 : index
          %26 = arith.addi %25, %arg3 : index
          %27 = arith.cmpi slt, %26, %dim_34 : index
          %28 = scf.if %27 -> (f32) {
            %29 = memref.load %alloc_8[%26, %21] : memref<?x1xf32, #gpu.address_space<global>>
            %30 = arith.addf %arg4, %29 : f32
            scf.yield %30 : f32
          } else {
            scf.yield %arg4 : f32
          }
          scf.yield %28 : f32
        }
        scf.yield %24 : f32
      } else {
        scf.yield %17 : f32
      }
      scf.if %22 {
        %24 = memref.atomic_rmw addf %23, %alloc_9[%21] : (f32, memref<1xf32, #gpu.address_space<global>>) -> f32
      }
      scf.yield
    }
    "lmhlo.terminator"() : () -> ()
  }) {disc.device = "gpu", disc.fusion.name = "main_kColReduction_reduce__6_1_0", disc.fusion_type = "kColReduction"} : () -> ()
  memref.dealloc %alloc_8 : memref<?x1xf32, #gpu.address_space<global>>
  memref.dealloc %alloc_7 : memref<?x4xf32, #gpu.address_space<global>>
  memref.dealloc %alloc_6 : memref<?x4xf32, #gpu.address_space<global>>
  memref.dealloc %alloc_5 : memref<?x4xf32, #gpu.address_space<global>>
  memref.dealloc %alloc_4 : memref<f32, #gpu.address_space<global>>
  memref.dealloc %alloc_2 : memref<?x4xf32, #gpu.address_space<global>>
  memref.dealloc %alloc : memref<?x4xf32, #gpu.address_space<global>>
  %10 = llvm.inttoptr %0 : i32 to !llvm.ptr<i8>
  %alloca_24 = memref.alloca() : memref<0xindex>
  %11 = "disc_ral.dispatch"(%arg0, %10, %alloc_9, %alloca_24) {backend_config = "", call_target_name = "inc_ref", device = "gpu", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>, memref<1xf32, #gpu.address_space<global>>, memref<0xindex>) -> memref<f32, #gpu.address_space<global>>
  %reinterpret_cast_25 = memref.reinterpret_cast %11 to offset: [0], sizes: [], strides: [] : memref<f32, #gpu.address_space<global>> to memref<f32, #gpu.address_space<global>>
  memref.dealloc %alloc_9 : memref<1xf32, #gpu.address_space<global>>
  %alloc_26 = memref.alloc() : memref<f32>
  %12 = llvm.inttoptr %0 : i32 to !llvm.ptr<i8>
  "disc_ral.dispatch"(%arg0, %12, %reinterpret_cast_25, %alloc_26) {backend_config = "", call_target_name = "d2h", device = "gpu", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>, memref<f32, #gpu.address_space<global>>, memref<f32>) -> ()
  "disc_ral.dispatch"(%arg0, %12) {backend_config = "", call_target_name = "sync_on_stream", device = "gpu", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>) -> ()
  memref.dealloc %reinterpret_cast_25 : memref<f32, #gpu.address_space<global>>
  "disc_ral.dispatch"(%arg0, %c0, %alloc_26) {backend_config = "", call_target_name = "ral_send_output", device = "cpu", has_side_effect = false} : (!disc_ral.context, index, memref<f32>) -> ()
  return
}

