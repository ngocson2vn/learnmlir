module {
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
    %10 = arith.cmpi eq, %dim_0, %4 : index
    %11 = arith.cmpi eq, %dim, %4 : index
    %12 = arith.andi %11, %10 : i1
    %alloc_4 = memref.alloc() : memref<f32, #gpu.address_space<global>>
    %alloc_5 = memref.alloc(%4) {kDiscSymbolicDimAttr = [@S2, @C4]} : memref<?x4xf32, #gpu.address_space<global>>
    %alloc_6 = memref.alloc(%4) {kDiscSymbolicDimAttr = [@S2, @C4]} : memref<?x4xf32, #gpu.address_space<global>>
    %alloc_7 = memref.alloc(%4) {kDiscSymbolicDimAttr = [@S2, @C4]} : memref<?x4xf32, #gpu.address_space<global>>
    %alloc_8 = memref.alloc(%9) {kDiscSymbolicDimAttr = [@S3, @C1]} : memref<?x1xf32, #gpu.address_space<global>>
    %alloc_9 = memref.alloc() : memref<1xf32, #gpu.address_space<global>>
    scf.if %12 {
      %reinterpret_cast_13 = memref.reinterpret_cast %alloc to offset: [0], sizes: [%4, 4], strides: [4, 1] {kDiscSymbolicDimAttr = [@S4, @C4]} : memref<?x4xf32, #gpu.address_space<global>> to memref<?x4xf32, #gpu.address_space<global>>
      %reinterpret_cast_14 = memref.reinterpret_cast %alloc_2 to offset: [0], sizes: [%4, 4], strides: [4, 1] {kDiscSymbolicDimAttr = [@S4, @C4]} : memref<?x4xf32, #gpu.address_space<global>> to memref<?x4xf32, #gpu.address_space<global>>
      %reinterpret_cast_15 = memref.reinterpret_cast %alloc_7 to offset: [0], sizes: [%4, 4], strides: [4, 1] {kDiscSymbolicDimAttr = [@S4, @C4]} : memref<?x4xf32, #gpu.address_space<global>> to memref<?x4xf32, #gpu.address_space<global>>
      %reinterpret_cast_16 = memref.reinterpret_cast %alloc_8 to offset: [0], sizes: [%9, 1], strides: [1, 1] {kDiscSymbolicDimAttr = [@S5, @C1]} : memref<?x1xf32, #gpu.address_space<global>> to memref<?x1xf32, #gpu.address_space<global>>
      %16 = arith.cmpi slt, %9, %c1 : index
      scf.if %16 {
        "lmhlo.fusion"() ({
          "lmhlo.constant"(%alloc_4) {disc.device = "gpu", value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, #gpu.address_space<global>>) -> ()
          "lmhlo.multiply"(%reinterpret_cast_13, %reinterpret_cast_14, %reinterpret_cast_15) {disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<?x4xf32, #gpu.address_space<global>>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
          "lmhlo.dynamic_reshape"(%reinterpret_cast_15, %alloca_3, %reinterpret_cast_16) {disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<2xi32>, memref<?x1xf32, #gpu.address_space<global>>) -> ()
          %c0_17 = arith.constant 0 : index
          %c1_18 = arith.constant 1 : index
          %c1_19 = arith.constant 1 : index
          %c0_20 = arith.constant 0 : index
          %dim_21 = memref.dim %alloc_9, %c0_20 : memref<1xf32, #gpu.address_space<global>>
          %17 = arith.muli %c1_19, %dim_21 : index
          scf.parallel (%arg1) = (%c0_17) to (%17) step (%c1_18) {
            %c1_26 = arith.constant 1 : index
            %21 = "disc_shape.delinearize"(%arg1, %c1_26) : (index, index) -> index
            %22 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            memref.store %22, %alloc_9[%21] : memref<1xf32, #gpu.address_space<global>>
            scf.yield
          }
          %c0_22 = arith.constant 0 : index
          %c1_23 = arith.constant 1 : index
          %dim_24 = memref.dim %reinterpret_cast_16, %c0_22 : memref<?x1xf32, #gpu.address_space<global>>
          %dim_25 = memref.dim %reinterpret_cast_16, %c1_23 : memref<?x1xf32, #gpu.address_space<global>>
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %18 = arith.ceildivui %dim_25, %c512 : index
          %19 = arith.ceildivui %dim_24, %c32 : index
          %20 = arith.muli %18, %19 : index
          scf.parallel (%arg1, %arg2) = (%c0_22, %c0_22) to (%20, %c512) step (%c1_23, %c1_23) {
            %21 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            %22 = arith.divui %arg1, %18 : index
            %23 = arith.remui %arg1, %18 : index
            %24 = arith.muli %23, %c512 : index
            %25 = arith.addi %24, %arg2 : index
            %26 = arith.cmpi ult, %25, %dim_25 : index
            %27 = scf.if %26 -> (f32) {
              %28 = scf.for %arg3 = %c0_22 to %c32 step %c1_23 iter_args(%arg4 = %21) -> (f32) {
                %29 = arith.muli %22, %c32 : index
                %30 = arith.addi %29, %arg3 : index
                %31 = arith.cmpi slt, %30, %dim_24 : index
                %32 = scf.if %31 -> (f32) {
                  %33 = memref.load %reinterpret_cast_16[%30, %25] : memref<?x1xf32, #gpu.address_space<global>>
                  %34 = arith.addf %arg4, %33 : f32
                  scf.yield %34 : f32
                } else {
                  scf.yield %arg4 : f32
                }
                scf.yield %32 : f32
              }
              scf.yield %28 : f32
            } else {
              scf.yield %21 : f32
            }
            scf.if %26 {
              %28 = memref.atomic_rmw addf %27, %alloc_9[%25] : (f32, memref<1xf32, #gpu.address_space<global>>) -> f32
            }
            scf.yield
          }
          "lmhlo.terminator"() : () -> ()
        }) {disc.device = "gpu", disc.fusion.name = "main_kColReduction_reduce__6_1_0", disc.fusion.tag = "no_ibXthread_tile_h32", disc.fusion_type = "kColReduction", disc_col_reduction_schedule_hint = 7 : i32, disc_cta_size_hint = 512 : i32} : () -> ()
      } else {
        "lmhlo.fusion"() ({
          "lmhlo.constant"(%alloc_4) {disc.device = "gpu", value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, #gpu.address_space<global>>) -> ()
          "lmhlo.multiply"(%reinterpret_cast_13, %reinterpret_cast_14, %reinterpret_cast_15) {disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<?x4xf32, #gpu.address_space<global>>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
          "lmhlo.dynamic_reshape"(%reinterpret_cast_15, %alloca_3, %reinterpret_cast_16) {disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<2xi32>, memref<?x1xf32, #gpu.address_space<global>>) -> ()
          %c0_17 = arith.constant 0 : index
          %c1_18 = arith.constant 1 : index
          %c1_19 = arith.constant 1 : index
          %c0_20 = arith.constant 0 : index
          %dim_21 = memref.dim %alloc_9, %c0_20 : memref<1xf32, #gpu.address_space<global>>
          %17 = arith.muli %c1_19, %dim_21 : index
          scf.parallel (%arg1) = (%c0_17) to (%17) step (%c1_18) {
            %c1_26 = arith.constant 1 : index
            %21 = "disc_shape.delinearize"(%arg1, %c1_26) : (index, index) -> index
            %22 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            memref.store %22, %alloc_9[%21] : memref<1xf32, #gpu.address_space<global>>
            scf.yield
          }
          %c0_22 = arith.constant 0 : index
          %c1_23 = arith.constant 1 : index
          %dim_24 = memref.dim %reinterpret_cast_16, %c0_22 : memref<?x1xf32, #gpu.address_space<global>>
          %dim_25 = memref.dim %reinterpret_cast_16, %c1_23 : memref<?x1xf32, #gpu.address_space<global>>
          %c32 = arith.constant 32 : index
          %c8 = arith.constant 8 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %c256 = arith.constant 256 : index
          %18 = arith.ceildivui %dim_25, %c32 : index
          %19 = arith.ceildivui %dim_24, %c512 : index
          %20 = arith.muli %18, %19 : index
          scf.parallel (%arg1, %arg2) = (%c0_22, %c0_22) to (%20, %c256) step (%c1_23, %c1_23) {
            %21 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            %22 = arith.divui %arg1, %18 : index
            %23 = arith.remui %arg1, %18 : index
            %24 = arith.divui %arg2, %c32 : index
            %25 = arith.remui %arg2, %c32 : index
            %26 = arith.muli %25, %c8 : index
            %27 = arith.addi %24, %26 : index
            %28 = arith.muli %23, %c32 : index
            %29 = arith.addi %25, %28 : index
            %alloc_26 = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
            %30 = arith.cmpi ult, %29, %dim_25 : index
            %31 = scf.if %30 -> (f32) {
              %37 = scf.for %arg3 = %c0_22 to %c64 step %c1_23 iter_args(%arg4 = %21) -> (f32) {
                %38 = arith.muli %22, %c8 : index
                %39 = arith.addi %24, %38 : index
                %40 = arith.muli %c64, %39 : index
                %41 = arith.addi %arg3, %40 : index
                %42 = arith.cmpi slt, %41, %dim_24 : index
                %43 = scf.if %42 -> (f32) {
                  %44 = memref.load %reinterpret_cast_16[%41, %29] : memref<?x1xf32, #gpu.address_space<global>>
                  %45 = arith.addf %arg4, %44 : f32
                  scf.yield %45 : f32
                } else {
                  scf.yield %arg4 : f32
                }
                scf.yield %43 : f32
              }
              scf.yield %37 : f32
            } else {
              scf.yield %21 : f32
            }
            memref.store %31, %alloc_26[%27] : memref<256xf32, #gpu.address_space<workgroup>>
            gpu.barrier
            %c4_27 = arith.constant 4 : index
            %32 = arith.cmpi slt, %24, %c4_27 : index
            scf.if %32 {
              %37 = arith.addi %27, %c4_27 : index
              %38 = memref.load %alloc_26[%27] : memref<256xf32, #gpu.address_space<workgroup>>
              %39 = memref.load %alloc_26[%37] : memref<256xf32, #gpu.address_space<workgroup>>
              %40 = arith.addf %38, %39 : f32
              memref.store %40, %alloc_26[%27] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %c2 = arith.constant 2 : index
            %33 = arith.cmpi slt, %24, %c2 : index
            scf.if %33 {
              %37 = arith.addi %27, %c2 : index
              %38 = memref.load %alloc_26[%27] : memref<256xf32, #gpu.address_space<workgroup>>
              %39 = memref.load %alloc_26[%37] : memref<256xf32, #gpu.address_space<workgroup>>
              %40 = arith.addf %38, %39 : f32
              memref.store %40, %alloc_26[%27] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %c1_28 = arith.constant 1 : index
            %34 = arith.cmpi slt, %24, %c1_28 : index
            scf.if %34 {
              %37 = arith.addi %27, %c1_28 : index
              %38 = memref.load %alloc_26[%27] : memref<256xf32, #gpu.address_space<workgroup>>
              %39 = memref.load %alloc_26[%37] : memref<256xf32, #gpu.address_space<workgroup>>
              %40 = arith.addf %38, %39 : f32
              memref.store %40, %alloc_26[%27] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %35 = arith.cmpi eq, %24, %c0_22 : index
            %36 = arith.andi %35, %30 : i1
            scf.if %36 {
              %37 = arith.addi %27, %c1_23 : index
              %38 = memref.load %alloc_26[%27] : memref<256xf32, #gpu.address_space<workgroup>>
              %39 = memref.atomic_rmw addf %38, %alloc_9[%29] : (f32, memref<1xf32, #gpu.address_space<global>>) -> f32
            }
            scf.yield
          }
          "lmhlo.terminator"() : () -> ()
        }) {disc.device = "gpu", disc.fusion.name = "main_kColReduction_reduce__6_1_0", disc.fusion.tag = "no_ibXblock_tile_h64", disc.fusion_type = "kColReduction", disc_col_reduction_schedule_hint = 8 : i32, disc_cta_size_hint = 256 : i32} : () -> ()
      }
    } else {
      %16 = arith.cmpi slt, %9, %c1 : index
      scf.if %16 {
        "lmhlo.fusion"() ({
          "lmhlo.constant"(%alloc_4) {disc.device = "gpu", value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, #gpu.address_space<global>>) -> ()
          "lmhlo.dynamic_broadcast_in_dim"(%alloc, %alloca, %alloc_5) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>, disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<2xindex>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
          "lmhlo.dynamic_broadcast_in_dim"(%alloc_2, %alloca, %alloc_6) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>, disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<2xindex>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
          "lmhlo.multiply"(%alloc_5, %alloc_6, %alloc_7) {disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<?x4xf32, #gpu.address_space<global>>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
          "lmhlo.dynamic_reshape"(%alloc_7, %alloca_3, %alloc_8) {disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<2xi32>, memref<?x1xf32, #gpu.address_space<global>>) -> ()
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          %c1_15 = arith.constant 1 : index
          %c0_16 = arith.constant 0 : index
          %dim_17 = memref.dim %alloc_9, %c0_16 : memref<1xf32, #gpu.address_space<global>>
          %17 = arith.muli %c1_15, %dim_17 : index
          scf.parallel (%arg1) = (%c0_13) to (%17) step (%c1_14) {
            %c1_22 = arith.constant 1 : index
            %21 = "disc_shape.delinearize"(%arg1, %c1_22) : (index, index) -> index
            %22 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            memref.store %22, %alloc_9[%21] : memref<1xf32, #gpu.address_space<global>>
            scf.yield
          }
          %c0_18 = arith.constant 0 : index
          %c1_19 = arith.constant 1 : index
          %dim_20 = memref.dim %alloc_8, %c0_18 : memref<?x1xf32, #gpu.address_space<global>>
          %dim_21 = memref.dim %alloc_8, %c1_19 : memref<?x1xf32, #gpu.address_space<global>>
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %18 = arith.ceildivui %dim_21, %c512 : index
          %19 = arith.ceildivui %dim_20, %c32 : index
          %20 = arith.muli %18, %19 : index
          scf.parallel (%arg1, %arg2) = (%c0_18, %c0_18) to (%20, %c512) step (%c1_19, %c1_19) {
            %21 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            %22 = arith.divui %arg1, %18 : index
            %23 = arith.remui %arg1, %18 : index
            %24 = arith.muli %23, %c512 : index
            %25 = arith.addi %24, %arg2 : index
            %26 = arith.cmpi ult, %25, %dim_21 : index
            %27 = scf.if %26 -> (f32) {
              %28 = scf.for %arg3 = %c0_18 to %c32 step %c1_19 iter_args(%arg4 = %21) -> (f32) {
                %29 = arith.muli %22, %c32 : index
                %30 = arith.addi %29, %arg3 : index
                %31 = arith.cmpi slt, %30, %dim_20 : index
                %32 = scf.if %31 -> (f32) {
                  %33 = memref.load %alloc_8[%30, %25] : memref<?x1xf32, #gpu.address_space<global>>
                  %34 = arith.addf %arg4, %33 : f32
                  scf.yield %34 : f32
                } else {
                  scf.yield %arg4 : f32
                }
                scf.yield %32 : f32
              }
              scf.yield %28 : f32
            } else {
              scf.yield %21 : f32
            }
            scf.if %26 {
              %28 = memref.atomic_rmw addf %27, %alloc_9[%25] : (f32, memref<1xf32, #gpu.address_space<global>>) -> f32
            }
            scf.yield
          }
          "lmhlo.terminator"() : () -> ()
        }) {disc.device = "gpu", disc.fusion.name = "main_kColReduction_reduce__6_1_0", disc.fusion.tag = "thread_tile_h32", disc.fusion_type = "kColReduction", disc_col_reduction_schedule_hint = 7 : i32, disc_cta_size_hint = 512 : i32} : () -> ()
      } else {
        "lmhlo.fusion"() ({
          "lmhlo.constant"(%alloc_4) {disc.device = "gpu", value = dense<0.000000e+00> : tensor<f32>} : (memref<f32, #gpu.address_space<global>>) -> ()
          "lmhlo.dynamic_broadcast_in_dim"(%alloc, %alloca, %alloc_5) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>, disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<2xindex>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
          "lmhlo.dynamic_broadcast_in_dim"(%alloc_2, %alloca, %alloc_6) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>, disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<2xindex>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
          "lmhlo.multiply"(%alloc_5, %alloc_6, %alloc_7) {disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<?x4xf32, #gpu.address_space<global>>, memref<?x4xf32, #gpu.address_space<global>>) -> ()
          "lmhlo.dynamic_reshape"(%alloc_7, %alloca_3, %alloc_8) {disc.device = "gpu"} : (memref<?x4xf32, #gpu.address_space<global>>, memref<2xi32>, memref<?x1xf32, #gpu.address_space<global>>) -> ()
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          %c1_15 = arith.constant 1 : index
          %c0_16 = arith.constant 0 : index
          %dim_17 = memref.dim %alloc_9, %c0_16 : memref<1xf32, #gpu.address_space<global>>
          %17 = arith.muli %c1_15, %dim_17 : index
          scf.parallel (%arg1) = (%c0_13) to (%17) step (%c1_14) {
            %c1_22 = arith.constant 1 : index
            %21 = "disc_shape.delinearize"(%arg1, %c1_22) : (index, index) -> index
            %22 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            memref.store %22, %alloc_9[%21] : memref<1xf32, #gpu.address_space<global>>
            scf.yield
          }
          %c0_18 = arith.constant 0 : index
          %c1_19 = arith.constant 1 : index
          %dim_20 = memref.dim %alloc_8, %c0_18 : memref<?x1xf32, #gpu.address_space<global>>
          %dim_21 = memref.dim %alloc_8, %c1_19 : memref<?x1xf32, #gpu.address_space<global>>
          %c32 = arith.constant 32 : index
          %c8 = arith.constant 8 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %c256 = arith.constant 256 : index
          %18 = arith.ceildivui %dim_21, %c32 : index
          %19 = arith.ceildivui %dim_20, %c512 : index
          %20 = arith.muli %18, %19 : index
          scf.parallel (%arg1, %arg2) = (%c0_18, %c0_18) to (%20, %c256) step (%c1_19, %c1_19) {
            %21 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            %22 = arith.divui %arg1, %18 : index
            %23 = arith.remui %arg1, %18 : index
            %24 = arith.divui %arg2, %c32 : index
            %25 = arith.remui %arg2, %c32 : index
            %26 = arith.muli %25, %c8 : index
            %27 = arith.addi %24, %26 : index
            %28 = arith.muli %23, %c32 : index
            %29 = arith.addi %25, %28 : index
            %alloc_22 = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
            %30 = arith.cmpi ult, %29, %dim_21 : index
            %31 = scf.if %30 -> (f32) {
              %37 = scf.for %arg3 = %c0_18 to %c64 step %c1_19 iter_args(%arg4 = %21) -> (f32) {
                %38 = arith.muli %22, %c8 : index
                %39 = arith.addi %24, %38 : index
                %40 = arith.muli %c64, %39 : index
                %41 = arith.addi %arg3, %40 : index
                %42 = arith.cmpi slt, %41, %dim_20 : index
                %43 = scf.if %42 -> (f32) {
                  %44 = memref.load %alloc_8[%41, %29] : memref<?x1xf32, #gpu.address_space<global>>
                  %45 = arith.addf %arg4, %44 : f32
                  scf.yield %45 : f32
                } else {
                  scf.yield %arg4 : f32
                }
                scf.yield %43 : f32
              }
              scf.yield %37 : f32
            } else {
              scf.yield %21 : f32
            }
            memref.store %31, %alloc_22[%27] : memref<256xf32, #gpu.address_space<workgroup>>
            gpu.barrier
            %c4_23 = arith.constant 4 : index
            %32 = arith.cmpi slt, %24, %c4_23 : index
            scf.if %32 {
              %37 = arith.addi %27, %c4_23 : index
              %38 = memref.load %alloc_22[%27] : memref<256xf32, #gpu.address_space<workgroup>>
              %39 = memref.load %alloc_22[%37] : memref<256xf32, #gpu.address_space<workgroup>>
              %40 = arith.addf %38, %39 : f32
              memref.store %40, %alloc_22[%27] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %c2 = arith.constant 2 : index
            %33 = arith.cmpi slt, %24, %c2 : index
            scf.if %33 {
              %37 = arith.addi %27, %c2 : index
              %38 = memref.load %alloc_22[%27] : memref<256xf32, #gpu.address_space<workgroup>>
              %39 = memref.load %alloc_22[%37] : memref<256xf32, #gpu.address_space<workgroup>>
              %40 = arith.addf %38, %39 : f32
              memref.store %40, %alloc_22[%27] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %c1_24 = arith.constant 1 : index
            %34 = arith.cmpi slt, %24, %c1_24 : index
            scf.if %34 {
              %37 = arith.addi %27, %c1_24 : index
              %38 = memref.load %alloc_22[%27] : memref<256xf32, #gpu.address_space<workgroup>>
              %39 = memref.load %alloc_22[%37] : memref<256xf32, #gpu.address_space<workgroup>>
              %40 = arith.addf %38, %39 : f32
              memref.store %40, %alloc_22[%27] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %35 = arith.cmpi eq, %24, %c0_18 : index
            %36 = arith.andi %35, %30 : i1
            scf.if %36 {
              %37 = arith.addi %27, %c1_19 : index
              %38 = memref.load %alloc_22[%27] : memref<256xf32, #gpu.address_space<workgroup>>
              %39 = memref.atomic_rmw addf %38, %alloc_9[%29] : (f32, memref<1xf32, #gpu.address_space<global>>) -> f32
            }
            scf.yield
          }
          "lmhlo.terminator"() : () -> ()
        }) {disc.device = "gpu", disc.fusion.name = "main_kColReduction_reduce__6_1_0", disc.fusion.tag = "block_tile_h64", disc.fusion_type = "kColReduction", disc_col_reduction_schedule_hint = 8 : i32, disc_cta_size_hint = 256 : i32} : () -> ()
      }
    }
    memref.dealloc %alloc_8 : memref<?x1xf32, #gpu.address_space<global>>
    memref.dealloc %alloc_7 : memref<?x4xf32, #gpu.address_space<global>>
    memref.dealloc %alloc_6 : memref<?x4xf32, #gpu.address_space<global>>
    memref.dealloc %alloc_5 : memref<?x4xf32, #gpu.address_space<global>>
    memref.dealloc %alloc_4 : memref<f32, #gpu.address_space<global>>
    memref.dealloc %alloc_2 : memref<?x4xf32, #gpu.address_space<global>>
    memref.dealloc %alloc : memref<?x4xf32, #gpu.address_space<global>>
    %13 = llvm.inttoptr %0 : i32 to !llvm.ptr<i8>
    %alloca_10 = memref.alloca() : memref<0xindex>
    %14 = "disc_ral.dispatch"(%arg0, %13, %alloc_9, %alloca_10) {backend_config = "", call_target_name = "inc_ref", device = "gpu", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>, memref<1xf32, #gpu.address_space<global>>, memref<0xindex>) -> memref<f32, #gpu.address_space<global>>
    %reinterpret_cast_11 = memref.reinterpret_cast %14 to offset: [0], sizes: [], strides: [] : memref<f32, #gpu.address_space<global>> to memref<f32, #gpu.address_space<global>>
    memref.dealloc %alloc_9 : memref<1xf32, #gpu.address_space<global>>
    %alloc_12 = memref.alloc() : memref<f32>
    %15 = llvm.inttoptr %0 : i32 to !llvm.ptr<i8>
    "disc_ral.dispatch"(%arg0, %15, %reinterpret_cast_11, %alloc_12) {backend_config = "", call_target_name = "d2h", device = "gpu", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>, memref<f32, #gpu.address_space<global>>, memref<f32>) -> ()
    "disc_ral.dispatch"(%arg0, %15) {backend_config = "", call_target_name = "sync_on_stream", device = "gpu", has_side_effect = false} : (!disc_ral.context, !llvm.ptr<i8>) -> ()
    memref.dealloc %reinterpret_cast_11 : memref<f32, #gpu.address_space<global>>
    "disc_ral.dispatch"(%arg0, %c0, %alloc_12) {backend_config = "", call_target_name = "ral_send_output", device = "cpu", has_side_effect = false} : (!disc_ral.context, index, memref<f32>) -> ()
    return
  }
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S0", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = true, knownNonSizeZero = true, sym_name = "C4", value = 4 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S1", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S2", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = true, sym_name = "C1", value = 1 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = true, knownNonSizeZero = false, sym_name = "S3", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = false, knownNonSizeZero = false, sym_name = "S4", value = -9223372036854775808 : i64} : () -> ()
  "disc_shape.SymbolicDim"() {knownNegativeOne = false, knownNonNegative = true, knownNonSizeOne = true, knownNonSizeZero = false, sym_name = "S5", value = -9223372036854775808 : i64} : () -> ()
  func.func @shape_constraint_graph() {
    %c4 = arith.constant 4 : index
    %0 = "disc_shape.dim"() {name = @S2} : () -> index
    %1 = "disc_shape.dim"() {name = @S3} : () -> index
    "disc_shape.tie_product_equal"(%c4, %0, %1) {operand_segment_sizes = array<i32: 2, 1>} : (index, index, index) -> ()
    %c4_0 = arith.constant 4 : index
    %2 = "disc_shape.dim"() {name = @S4} : () -> index
    %3 = "disc_shape.dim"() {name = @S5} : () -> index
    "disc_shape.tie_product_equal"(%c4_0, %2, %3) {operand_segment_sizes = array<i32: 2, 1>} : (index, index, index) -> ()
    return
  }
}
