module {
  func.func @main(%arg0: !disc_ral.context) attributes {tf.entry_function = {input_placements = "cpu,cpu", inputs = "arg0,arg1", output_placements = "cpu", outputs = "out0"}} {
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
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
          scf.parallel (%arg1) = (%c0) to (%c1) step (%c1) {
            %18 = "disc_shape.delinearize"(%arg1, %c1) : (index, index) -> index
            %19 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            memref.store %19, %alloc_9[%18] : memref<1xf32, #gpu.address_space<global>>
            scf.yield
          }
          %17 = arith.ceildivui %9, %c32 : index
          scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%17, %c512) step (%c1, %c1) {
            %18 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            %19 = arith.cmpi ult, %arg2, %c1 : index
            %20 = scf.if %19 -> (f32) {
              %21 = scf.for %arg3 = %c0 to %c32 step %c1 iter_args(%arg4 = %18) -> (f32) {
                %22 = arith.muli %arg1, %c32 : index
                %23 = arith.addi %22, %arg3 : index
                %24 = arith.cmpi slt, %23, %9 : index
                %25 = scf.if %24 -> (f32) {
                  %26 = memref.load %reinterpret_cast_16[%23, %arg2] : memref<?x1xf32, #gpu.address_space<global>>
                  %27 = arith.addf %arg4, %26 : f32
                  scf.yield %27 : f32
                } else {
                  scf.yield %arg4 : f32
                }
                scf.yield %25 : f32
              }
              scf.yield %21 : f32
            } else {
              scf.yield %18 : f32
            }
            scf.if %19 {
              %21 = memref.atomic_rmw addf %20, %alloc_9[%arg2] : (f32, memref<1xf32, #gpu.address_space<global>>) -> f32
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
          scf.parallel (%arg1) = (%c0) to (%c1) step (%c1) {
            %18 = "disc_shape.delinearize"(%arg1, %c1) : (index, index) -> index
            %19 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            memref.store %19, %alloc_9[%18] : memref<1xf32, #gpu.address_space<global>>
            scf.yield
          }
          %17 = arith.ceildivui %9, %c512 : index
          scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%17, %c256) step (%c1, %c1) {
            %18 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            %19 = arith.divui %arg2, %c32 : index
            %20 = arith.remui %arg2, %c32 : index
            %21 = arith.muli %20, %c8 : index
            %22 = arith.addi %19, %21 : index
            %alloc_17 = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
            %23 = arith.cmpi ult, %20, %c1 : index
            %24 = scf.if %23 -> (f32) {
              %30 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %18) -> (f32) {
                %31 = arith.muli %arg1, %c8 : index
                %32 = arith.addi %19, %31 : index
                %33 = arith.muli %32, %c64 : index
                %34 = arith.addi %arg3, %33 : index
                %35 = arith.cmpi slt, %34, %9 : index
                %36 = scf.if %35 -> (f32) {
                  %37 = memref.load %reinterpret_cast_16[%34, %20] : memref<?x1xf32, #gpu.address_space<global>>
                  %38 = arith.addf %arg4, %37 : f32
                  scf.yield %38 : f32
                } else {
                  scf.yield %arg4 : f32
                }
                scf.yield %36 : f32
              }
              scf.yield %30 : f32
            } else {
              scf.yield %18 : f32
            }
            memref.store %24, %alloc_17[%22] : memref<256xf32, #gpu.address_space<workgroup>>
            gpu.barrier
            %25 = arith.cmpi slt, %19, %c4 : index
            scf.if %25 {
              %30 = arith.addi %22, %c4 : index
              %31 = memref.load %alloc_17[%22] : memref<256xf32, #gpu.address_space<workgroup>>
              %32 = memref.load %alloc_17[%30] : memref<256xf32, #gpu.address_space<workgroup>>
              %33 = arith.addf %31, %32 : f32
              memref.store %33, %alloc_17[%22] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %26 = arith.cmpi slt, %19, %c2 : index
            scf.if %26 {
              %30 = arith.addi %22, %c2 : index
              %31 = memref.load %alloc_17[%22] : memref<256xf32, #gpu.address_space<workgroup>>
              %32 = memref.load %alloc_17[%30] : memref<256xf32, #gpu.address_space<workgroup>>
              %33 = arith.addf %31, %32 : f32
              memref.store %33, %alloc_17[%22] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %27 = arith.cmpi slt, %19, %c1 : index
            scf.if %27 {
              %30 = arith.addi %22, %c1 : index
              %31 = memref.load %alloc_17[%22] : memref<256xf32, #gpu.address_space<workgroup>>
              %32 = memref.load %alloc_17[%30] : memref<256xf32, #gpu.address_space<workgroup>>
              %33 = arith.addf %31, %32 : f32
              memref.store %33, %alloc_17[%22] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %28 = arith.cmpi eq, %19, %c0 : index
            %29 = arith.andi %28, %23 : i1
            scf.if %29 {
              %30 = memref.load %alloc_17[%22] : memref<256xf32, #gpu.address_space<workgroup>>
              %31 = memref.atomic_rmw addf %30, %alloc_9[%20] : (f32, memref<1xf32, #gpu.address_space<global>>) -> f32
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
          scf.parallel (%arg1) = (%c0) to (%c1) step (%c1) {
            %18 = "disc_shape.delinearize"(%arg1, %c1) : (index, index) -> index
            %19 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            memref.store %19, %alloc_9[%18] : memref<1xf32, #gpu.address_space<global>>
            scf.yield
          }
          %17 = arith.ceildivui %9, %c32 : index
          scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%17, %c512) step (%c1, %c1) {
            %18 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            %19 = arith.cmpi ult, %arg2, %c1 : index
            %20 = scf.if %19 -> (f32) {
              %21 = scf.for %arg3 = %c0 to %c32 step %c1 iter_args(%arg4 = %18) -> (f32) {
                %22 = arith.muli %arg1, %c32 : index
                %23 = arith.addi %22, %arg3 : index
                %24 = arith.cmpi slt, %23, %9 : index
                %25 = scf.if %24 -> (f32) {
                  %26 = memref.load %alloc_8[%23, %arg2] : memref<?x1xf32, #gpu.address_space<global>>
                  %27 = arith.addf %arg4, %26 : f32
                  scf.yield %27 : f32
                } else {
                  scf.yield %arg4 : f32
                }
                scf.yield %25 : f32
              }
              scf.yield %21 : f32
            } else {
              scf.yield %18 : f32
            }
            scf.if %19 {
              %21 = memref.atomic_rmw addf %20, %alloc_9[%arg2] : (f32, memref<1xf32, #gpu.address_space<global>>) -> f32
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
          scf.parallel (%arg1) = (%c0) to (%c1) step (%c1) {
            %18 = "disc_shape.delinearize"(%arg1, %c1) : (index, index) -> index
            %19 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            memref.store %19, %alloc_9[%18] : memref<1xf32, #gpu.address_space<global>>
            scf.yield
          }
          %17 = arith.ceildivui %9, %c512 : index
          scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%17, %c256) step (%c1, %c1) {
            %18 = memref.load %alloc_4[] : memref<f32, #gpu.address_space<global>>
            %19 = arith.divui %arg2, %c32 : index
            %20 = arith.remui %arg2, %c32 : index
            %21 = arith.muli %20, %c8 : index
            %22 = arith.addi %19, %21 : index
            %alloc_13 = memref.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
            %23 = arith.cmpi ult, %20, %c1 : index
            %24 = scf.if %23 -> (f32) {
              %30 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %18) -> (f32) {
                %31 = arith.muli %arg1, %c8 : index
                %32 = arith.addi %19, %31 : index
                %33 = arith.muli %32, %c64 : index
                %34 = arith.addi %arg3, %33 : index
                %35 = arith.cmpi slt, %34, %9 : index
                %36 = scf.if %35 -> (f32) {
                  %37 = memref.load %alloc_8[%34, %20] : memref<?x1xf32, #gpu.address_space<global>>
                  %38 = arith.addf %arg4, %37 : f32
                  scf.yield %38 : f32
                } else {
                  scf.yield %arg4 : f32
                }
                scf.yield %36 : f32
              }
              scf.yield %30 : f32
            } else {
              scf.yield %18 : f32
            }
            memref.store %24, %alloc_13[%22] : memref<256xf32, #gpu.address_space<workgroup>>
            gpu.barrier
            %25 = arith.cmpi slt, %19, %c4 : index
            scf.if %25 {
              %30 = arith.addi %22, %c4 : index
              %31 = memref.load %alloc_13[%22] : memref<256xf32, #gpu.address_space<workgroup>>
              %32 = memref.load %alloc_13[%30] : memref<256xf32, #gpu.address_space<workgroup>>
              %33 = arith.addf %31, %32 : f32
              memref.store %33, %alloc_13[%22] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %26 = arith.cmpi slt, %19, %c2 : index
            scf.if %26 {
              %30 = arith.addi %22, %c2 : index
              %31 = memref.load %alloc_13[%22] : memref<256xf32, #gpu.address_space<workgroup>>
              %32 = memref.load %alloc_13[%30] : memref<256xf32, #gpu.address_space<workgroup>>
              %33 = arith.addf %31, %32 : f32
              memref.store %33, %alloc_13[%22] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %27 = arith.cmpi slt, %19, %c1 : index
            scf.if %27 {
              %30 = arith.addi %22, %c1 : index
              %31 = memref.load %alloc_13[%22] : memref<256xf32, #gpu.address_space<workgroup>>
              %32 = memref.load %alloc_13[%30] : memref<256xf32, #gpu.address_space<workgroup>>
              %33 = arith.addf %31, %32 : f32
              memref.store %33, %alloc_13[%22] : memref<256xf32, #gpu.address_space<workgroup>>
            }
            gpu.barrier
            %28 = arith.cmpi eq, %19, %c0 : index
            %29 = arith.andi %28, %23 : i1
            scf.if %29 {
              %30 = memref.load %alloc_13[%22] : memref<256xf32, #gpu.address_space<workgroup>>
              %31 = memref.atomic_rmw addf %30, %alloc_9[%20] : (f32, memref<1xf32, #gpu.address_space<global>>) -> f32
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
