// -----// IR Dump Before AFHandleFallbackPass (af-handle-fallback) //----- //
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %cst = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<>], device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<>], device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.Sum"(%arg0, %cst_0) {_symbolic_output_shapes = [#tf_type.shape<137>], device = "", keep_dims = false} : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
    %1 = "tf.GatherV2"(%arg1, %0, %cst) {_symbolic_output_shapes = [#tf_type.shape<137x4>], batch_dims = 0 : i64, device = ""} : (tensor<?x4xf32>, tensor<?xi32>, tensor<i32>) -> tensor<?x4xf32>
    %2 = "tf.Sum"(%arg2, %cst_0) {_symbolic_output_shapes = [#tf_type.shape<137>], device = "", keep_dims = false} : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
    %3 = "tf.GatherV2"(%arg3, %2, %cst) {_symbolic_output_shapes = [#tf_type.shape<137x4>], batch_dims = 0 : i64, device = ""} : (tensor<?x4xf32>, tensor<?xi32>, tensor<i32>) -> tensor<?x4xf32>
    %4 = "tf.Sum"(%arg4, %cst_0) {_symbolic_output_shapes = [#tf_type.shape<137>], device = "", keep_dims = false} : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
    %5 = "tf.GatherV2"(%arg5, %4, %cst) {_symbolic_output_shapes = [#tf_type.shape<137x4>], batch_dims = 0 : i64, device = ""} : (tensor<?x4xf32>, tensor<?xi32>, tensor<i32>) -> tensor<?x4xf32>
    return %1, %3, %5 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump After AFHandleFallbackPass (af-handle-fallback) //----- //
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %cst = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<>], device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<>], device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.Sum"(%arg0, %cst_0) {_symbolic_output_shapes = [#tf_type.shape<137>], device = "", keep_dims = false} : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
    %1 = "tf.GatherV2"(%arg1, %0, %cst) {_symbolic_output_shapes = [#tf_type.shape<137x4>], batch_dims = 0 : i64, device = ""} : (tensor<?x4xf32>, tensor<?xi32>, tensor<i32>) -> tensor<?x4xf32>
    %2 = "tf.Sum"(%arg2, %cst_0) {_symbolic_output_shapes = [#tf_type.shape<137>], device = "", keep_dims = false} : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
    %3 = "tf.GatherV2"(%arg3, %2, %cst) {_symbolic_output_shapes = [#tf_type.shape<137x4>], batch_dims = 0 : i64, device = ""} : (tensor<?x4xf32>, tensor<?xi32>, tensor<i32>) -> tensor<?x4xf32>
    %4 = "tf.Sum"(%arg4, %cst_0) {_symbolic_output_shapes = [#tf_type.shape<137>], device = "", keep_dims = false} : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
    %5 = "tf.GatherV2"(%arg5, %4, %cst) {_symbolic_output_shapes = [#tf_type.shape<137x4>], batch_dims = 0 : i64, device = ""} : (tensor<?x4xf32>, tensor<?xi32>, tensor<i32>) -> tensor<?x4xf32>
    return %1, %3, %5 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump Before LegalizeTFNoFallback (xla-legalize-tf-no-fallback) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %cst = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<>], device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_0 = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<>], device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Sum"(%arg0, %cst_0) {_symbolic_output_shapes = [#tf_type.shape<137>], device = "", keep_dims = false} : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %1 = "tf.GatherV2"(%arg1, %0, %cst) {_symbolic_output_shapes = [#tf_type.shape<137x4>], batch_dims = 0 : i64, device = ""} : (tensor<?x4xf32>, tensor<?xi32>, tensor<i32>) -> tensor<?x4xf32>
  %2 = "tf.Sum"(%arg2, %cst_0) {_symbolic_output_shapes = [#tf_type.shape<137>], device = "", keep_dims = false} : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %3 = "tf.GatherV2"(%arg3, %2, %cst) {_symbolic_output_shapes = [#tf_type.shape<137x4>], batch_dims = 0 : i64, device = ""} : (tensor<?x4xf32>, tensor<?xi32>, tensor<i32>) -> tensor<?x4xf32>
  %4 = "tf.Sum"(%arg4, %cst_0) {_symbolic_output_shapes = [#tf_type.shape<137>], device = "", keep_dims = false} : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %5 = "tf.GatherV2"(%arg5, %4, %cst) {_symbolic_output_shapes = [#tf_type.shape<137x4>], batch_dims = 0 : i64, device = ""} : (tensor<?x4xf32>, tensor<?xi32>, tensor<i32>) -> tensor<?x4xf32>
  return %1, %3, %5 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After LegalizeTFNoFallback (xla-legalize-tf-no-fallback) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  %2 = mhlo.convert %arg0 : tensor<?x2xi32>
  %3 = mhlo.constant dense<0> : tensor<i32>
  %4 = mhlo.reduce(%2 init: %3) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %5 = mhlo.convert %4 : tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg1, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %7 = mhlo.convert %arg2 : tensor<?x2xi32>
  %8 = mhlo.constant dense<0> : tensor<i32>
  %9 = mhlo.reduce(%7 init: %8) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %10 = mhlo.convert %9 : tensor<?xi32>
  %11 = "mhlo.torch_index_select"(%arg3, %10) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %12 = mhlo.convert %arg4 : tensor<?x2xi32>
  %13 = mhlo.constant dense<0> : tensor<i32>
  %14 = mhlo.reduce(%12 init: %13) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %15 = mhlo.convert %14 : tensor<?xi32>
  %16 = "mhlo.torch_index_select"(%arg5, %15) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %6, %11, %16 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before RankSpecializationClusterPass (mhlo-rank-specialization-cluster) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  %2 = mhlo.convert %arg0 : tensor<?x2xi32>
  %3 = mhlo.constant dense<0> : tensor<i32>
  %4 = mhlo.reduce(%2 init: %3) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %5 = mhlo.convert %4 : tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg1, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %7 = mhlo.convert %arg2 : tensor<?x2xi32>
  %8 = mhlo.constant dense<0> : tensor<i32>
  %9 = mhlo.reduce(%7 init: %8) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %10 = mhlo.convert %9 : tensor<?xi32>
  %11 = "mhlo.torch_index_select"(%arg3, %10) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %12 = mhlo.convert %arg4 : tensor<?x2xi32>
  %13 = mhlo.constant dense<0> : tensor<i32>
  %14 = mhlo.reduce(%12 init: %13) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %15 = mhlo.convert %14 : tensor<?xi32>
  %16 = "mhlo.torch_index_select"(%arg5, %15) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %6, %11, %16 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After RankSpecializationClusterPass (mhlo-rank-specialization-cluster) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before RankSpecializationToSCFPass (mhlo-rank-specialization-to-scf) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After RankSpecializationToSCFPass (mhlo-rank-specialization-to-scf) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before ChloLegalizeToHloPass (chlo-legalize-to-hlo) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After ChloLegalizeToHloPass (chlo-legalize-to-hlo) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before ShapeSimplification (shape-simplification) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After ShapeSimplification (shape-simplification) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before MergeAssumingLimitOps (merge-assuming-limit-ops) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After MergeAssumingLimitOps (merge-assuming-limit-ops) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before BroadcastPropagationPass (mhlo-broadcast-propagation) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After BroadcastPropagationPass (mhlo-broadcast-propagation) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before AddFakeSymbolicShape (add-fake-symbolic-shape) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After AddFakeSymbolicShape (add-fake-symbolic-shape) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before CanonicalizeExtWithConstraints (canonicalize-ext-with-constraints) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After CanonicalizeExtWithConstraints (canonicalize-ext-with-constraints) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before DelFakeSymbolicShape (del-fake-symbolic-shape) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After DelFakeSymbolicShape (del-fake-symbolic-shape) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before HloLegalizeToLinalgPass (hlo-legalize-to-linalg) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.reduce(%arg0 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %2 = "mhlo.torch_index_select"(%arg1, %1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %3 = mhlo.reduce(%arg2 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %4 = "mhlo.torch_index_select"(%arg3, %3) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  %5 = mhlo.reduce(%arg4 init: %0) applies mhlo.add across dimensions = [1] : (tensor<?x2xi32>, tensor<i32>) -> tensor<?xi32>
  %6 = "mhlo.torch_index_select"(%arg5, %5) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>
  return %2, %4, %6 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After HloLegalizeToLinalgPass (hlo-legalize-to-linalg) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %cst = arith.constant dense<0> : tensor<i32>
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_0 = arith.constant 0 : index
  %dim_1 = tensor.dim %2, %c0_0 : tensor<?xi32>
  %3 = tensor.empty() : tensor<4xf32>
  %4 = tensor.empty(%dim_1) : tensor<?x4xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?xi32>, tensor<4xf32>) outs(%4 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %c0_i32_2 = arith.constant 0 : i32
  %c0_3 = arith.constant 0 : index
  %dim_4 = tensor.dim %arg2, %c0_3 : tensor<?x2xi32>
  %6 = tensor.empty(%dim_4) : tensor<?xi32>
  %7 = linalg.fill ins(%c0_i32_2 : i32) outs(%6 : tensor<?xi32>) -> tensor<?xi32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%7 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_5 = arith.constant 0 : index
  %dim_6 = tensor.dim %8, %c0_5 : tensor<?xi32>
  %9 = tensor.empty() : tensor<4xf32>
  %10 = tensor.empty(%dim_6) : tensor<?x4xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %9 : tensor<?xi32>, tensor<4xf32>) outs(%10 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %c0_i32_7 = arith.constant 0 : i32
  %c0_8 = arith.constant 0 : index
  %dim_9 = tensor.dim %arg4, %c0_8 : tensor<?x2xi32>
  %12 = tensor.empty(%dim_9) : tensor<?xi32>
  %13 = linalg.fill ins(%c0_i32_7 : i32) outs(%12 : tensor<?xi32>) -> tensor<?xi32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%13 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_10 = arith.constant 0 : index
  %dim_11 = tensor.dim %14, %c0_10 : tensor<?xi32>
  %15 = tensor.empty() : tensor<4xf32>
  %16 = tensor.empty(%dim_11) : tensor<?x4xf32>
  %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<?xi32>, tensor<4xf32>) outs(%16 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %5, %11, %17 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before CustomHloLegalizeToLinalgPass (custom-hlo-legalize-to-linalg) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %cst = arith.constant dense<0> : tensor<i32>
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_0 = arith.constant 0 : index
  %dim_1 = tensor.dim %2, %c0_0 : tensor<?xi32>
  %3 = tensor.empty() : tensor<4xf32>
  %4 = tensor.empty(%dim_1) : tensor<?x4xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?xi32>, tensor<4xf32>) outs(%4 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %c0_i32_2 = arith.constant 0 : i32
  %c0_3 = arith.constant 0 : index
  %dim_4 = tensor.dim %arg2, %c0_3 : tensor<?x2xi32>
  %6 = tensor.empty(%dim_4) : tensor<?xi32>
  %7 = linalg.fill ins(%c0_i32_2 : i32) outs(%6 : tensor<?xi32>) -> tensor<?xi32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%7 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_5 = arith.constant 0 : index
  %dim_6 = tensor.dim %8, %c0_5 : tensor<?xi32>
  %9 = tensor.empty() : tensor<4xf32>
  %10 = tensor.empty(%dim_6) : tensor<?x4xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %9 : tensor<?xi32>, tensor<4xf32>) outs(%10 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %c0_i32_7 = arith.constant 0 : i32
  %c0_8 = arith.constant 0 : index
  %dim_9 = tensor.dim %arg4, %c0_8 : tensor<?x2xi32>
  %12 = tensor.empty(%dim_9) : tensor<?xi32>
  %13 = linalg.fill ins(%c0_i32_7 : i32) outs(%12 : tensor<?xi32>) -> tensor<?xi32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%13 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_10 = arith.constant 0 : index
  %dim_11 = tensor.dim %14, %c0_10 : tensor<?xi32>
  %15 = tensor.empty() : tensor<4xf32>
  %16 = tensor.empty(%dim_11) : tensor<?x4xf32>
  %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<?xi32>, tensor<4xf32>) outs(%16 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %5, %11, %17 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After CustomHloLegalizeToLinalgPass (custom-hlo-legalize-to-linalg) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %cst = arith.constant dense<0> : tensor<i32>
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_0 = arith.constant 0 : index
  %dim_1 = tensor.dim %2, %c0_0 : tensor<?xi32>
  %3 = tensor.empty() : tensor<4xf32>
  %4 = tensor.empty(%dim_1) : tensor<?x4xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?xi32>, tensor<4xf32>) outs(%4 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %c0_i32_2 = arith.constant 0 : i32
  %c0_3 = arith.constant 0 : index
  %dim_4 = tensor.dim %arg2, %c0_3 : tensor<?x2xi32>
  %6 = tensor.empty(%dim_4) : tensor<?xi32>
  %7 = linalg.fill ins(%c0_i32_2 : i32) outs(%6 : tensor<?xi32>) -> tensor<?xi32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%7 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_5 = arith.constant 0 : index
  %dim_6 = tensor.dim %8, %c0_5 : tensor<?xi32>
  %9 = tensor.empty() : tensor<4xf32>
  %10 = tensor.empty(%dim_6) : tensor<?x4xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %9 : tensor<?xi32>, tensor<4xf32>) outs(%10 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %c0_i32_7 = arith.constant 0 : i32
  %c0_8 = arith.constant 0 : index
  %dim_9 = tensor.dim %arg4, %c0_8 : tensor<?x2xi32>
  %12 = tensor.empty(%dim_9) : tensor<?xi32>
  %13 = linalg.fill ins(%c0_i32_7 : i32) outs(%12 : tensor<?xi32>) -> tensor<?xi32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%13 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_10 = arith.constant 0 : index
  %dim_11 = tensor.dim %14, %c0_10 : tensor<?xi32>
  %15 = tensor.empty() : tensor<4xf32>
  %16 = tensor.empty(%dim_11) : tensor<?x4xf32>
  %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<?xi32>, tensor<4xf32>) outs(%16 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %5, %11, %17 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before HloLegalizeToArithmeticPass (hlo-legalize-to-arithmetic) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %cst = arith.constant dense<0> : tensor<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_0 = arith.constant 0 : index
    %dim_1 = tensor.dim %2, %c0_0 : tensor<?xi32>
    %3 = tensor.empty() : tensor<4xf32>
    %4 = tensor.empty(%dim_1) : tensor<?x4xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?xi32>, tensor<4xf32>) outs(%4 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg1[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %c0_i32_2 = arith.constant 0 : i32
    %c0_3 = arith.constant 0 : index
    %dim_4 = tensor.dim %arg2, %c0_3 : tensor<?x2xi32>
    %6 = tensor.empty(%dim_4) : tensor<?xi32>
    %7 = linalg.fill ins(%c0_i32_2 : i32) outs(%6 : tensor<?xi32>) -> tensor<?xi32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%7 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_5 = arith.constant 0 : index
    %dim_6 = tensor.dim %8, %c0_5 : tensor<?xi32>
    %9 = tensor.empty() : tensor<4xf32>
    %10 = tensor.empty(%dim_6) : tensor<?x4xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %9 : tensor<?xi32>, tensor<4xf32>) outs(%10 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg3[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %c0_i32_7 = arith.constant 0 : i32
    %c0_8 = arith.constant 0 : index
    %dim_9 = tensor.dim %arg4, %c0_8 : tensor<?x2xi32>
    %12 = tensor.empty(%dim_9) : tensor<?xi32>
    %13 = linalg.fill ins(%c0_i32_7 : i32) outs(%12 : tensor<?xi32>) -> tensor<?xi32>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%13 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_10 = arith.constant 0 : index
    %dim_11 = tensor.dim %14, %c0_10 : tensor<?xi32>
    %15 = tensor.empty() : tensor<4xf32>
    %16 = tensor.empty(%dim_11) : tensor<?x4xf32>
    %17 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<?xi32>, tensor<4xf32>) outs(%16 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg5[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    return %5, %11, %17 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump After HloLegalizeToArithmeticPass (hlo-legalize-to-arithmetic) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %cst = arith.constant dense<0> : tensor<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_0 = arith.constant 0 : index
    %dim_1 = tensor.dim %2, %c0_0 : tensor<?xi32>
    %3 = tensor.empty() : tensor<4xf32>
    %4 = tensor.empty(%dim_1) : tensor<?x4xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?xi32>, tensor<4xf32>) outs(%4 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg1[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %c0_i32_2 = arith.constant 0 : i32
    %c0_3 = arith.constant 0 : index
    %dim_4 = tensor.dim %arg2, %c0_3 : tensor<?x2xi32>
    %6 = tensor.empty(%dim_4) : tensor<?xi32>
    %7 = linalg.fill ins(%c0_i32_2 : i32) outs(%6 : tensor<?xi32>) -> tensor<?xi32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%7 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_5 = arith.constant 0 : index
    %dim_6 = tensor.dim %8, %c0_5 : tensor<?xi32>
    %9 = tensor.empty() : tensor<4xf32>
    %10 = tensor.empty(%dim_6) : tensor<?x4xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %9 : tensor<?xi32>, tensor<4xf32>) outs(%10 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg3[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %c0_i32_7 = arith.constant 0 : i32
    %c0_8 = arith.constant 0 : index
    %dim_9 = tensor.dim %arg4, %c0_8 : tensor<?x2xi32>
    %12 = tensor.empty(%dim_9) : tensor<?xi32>
    %13 = linalg.fill ins(%c0_i32_7 : i32) outs(%12 : tensor<?xi32>) -> tensor<?xi32>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%13 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_10 = arith.constant 0 : index
    %dim_11 = tensor.dim %14, %c0_10 : tensor<?xi32>
    %15 = tensor.empty() : tensor<4xf32>
    %16 = tensor.empty(%dim_11) : tensor<?x4xf32>
    %17 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<?xi32>, tensor<4xf32>) outs(%16 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg5[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    return %5, %11, %17 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump Before HloLegalizeShapeOpsToStandardPass (hlo-legalize-shapeops-to-standard) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %cst = arith.constant dense<0> : tensor<i32>
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_0 = arith.constant 0 : index
  %dim_1 = tensor.dim %2, %c0_0 : tensor<?xi32>
  %3 = tensor.empty() : tensor<4xf32>
  %4 = tensor.empty(%dim_1) : tensor<?x4xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?xi32>, tensor<4xf32>) outs(%4 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %c0_i32_2 = arith.constant 0 : i32
  %c0_3 = arith.constant 0 : index
  %dim_4 = tensor.dim %arg2, %c0_3 : tensor<?x2xi32>
  %6 = tensor.empty(%dim_4) : tensor<?xi32>
  %7 = linalg.fill ins(%c0_i32_2 : i32) outs(%6 : tensor<?xi32>) -> tensor<?xi32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%7 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_5 = arith.constant 0 : index
  %dim_6 = tensor.dim %8, %c0_5 : tensor<?xi32>
  %9 = tensor.empty() : tensor<4xf32>
  %10 = tensor.empty(%dim_6) : tensor<?x4xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %9 : tensor<?xi32>, tensor<4xf32>) outs(%10 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %c0_i32_7 = arith.constant 0 : i32
  %c0_8 = arith.constant 0 : index
  %dim_9 = tensor.dim %arg4, %c0_8 : tensor<?x2xi32>
  %12 = tensor.empty(%dim_9) : tensor<?xi32>
  %13 = linalg.fill ins(%c0_i32_7 : i32) outs(%12 : tensor<?xi32>) -> tensor<?xi32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%13 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_10 = arith.constant 0 : index
  %dim_11 = tensor.dim %14, %c0_10 : tensor<?xi32>
  %15 = tensor.empty() : tensor<4xf32>
  %16 = tensor.empty(%dim_11) : tensor<?x4xf32>
  %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<?xi32>, tensor<4xf32>) outs(%16 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %5, %11, %17 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After HloLegalizeShapeOpsToStandardPass (hlo-legalize-shapeops-to-standard) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %cst = arith.constant dense<0> : tensor<i32>
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_0 = arith.constant 0 : index
  %dim_1 = tensor.dim %2, %c0_0 : tensor<?xi32>
  %3 = tensor.empty() : tensor<4xf32>
  %4 = tensor.empty(%dim_1) : tensor<?x4xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?xi32>, tensor<4xf32>) outs(%4 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %c0_i32_2 = arith.constant 0 : i32
  %c0_3 = arith.constant 0 : index
  %dim_4 = tensor.dim %arg2, %c0_3 : tensor<?x2xi32>
  %6 = tensor.empty(%dim_4) : tensor<?xi32>
  %7 = linalg.fill ins(%c0_i32_2 : i32) outs(%6 : tensor<?xi32>) -> tensor<?xi32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%7 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_5 = arith.constant 0 : index
  %dim_6 = tensor.dim %8, %c0_5 : tensor<?xi32>
  %9 = tensor.empty() : tensor<4xf32>
  %10 = tensor.empty(%dim_6) : tensor<?x4xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %9 : tensor<?xi32>, tensor<4xf32>) outs(%10 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %c0_i32_7 = arith.constant 0 : i32
  %c0_8 = arith.constant 0 : index
  %dim_9 = tensor.dim %arg4, %c0_8 : tensor<?x2xi32>
  %12 = tensor.empty(%dim_9) : tensor<?xi32>
  %13 = linalg.fill ins(%c0_i32_7 : i32) outs(%12 : tensor<?xi32>) -> tensor<?xi32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%13 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %from_elements = tensor.from_elements %out : tensor<i32>
    %from_elements_12 = tensor.from_elements %in : tensor<i32>
    %extracted = tensor.extract %from_elements[] : tensor<i32>
    %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
    %18 = arith.addi %extracted, %extracted_13 : i32
    %from_elements_14 = tensor.from_elements %18 : tensor<i32>
    %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
    linalg.yield %extracted_15 : i32
  } -> tensor<?xi32>
  %c0_10 = arith.constant 0 : index
  %dim_11 = tensor.dim %14, %c0_10 : tensor<?xi32>
  %15 = tensor.empty() : tensor<4xf32>
  %16 = tensor.empty(%dim_11) : tensor<?x4xf32>
  %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<?xi32>, tensor<4xf32>) outs(%16 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %in_12: f32, %out: f32):
    %18 = arith.index_cast %in : i32 to index
    %19 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%18, %19] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %5, %11, %17 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before ConvertToSignlessPass (convert-to-signless) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %cst = arith.constant dense<0> : tensor<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_0 = arith.constant 0 : index
    %dim_1 = tensor.dim %2, %c0_0 : tensor<?xi32>
    %3 = tensor.empty() : tensor<4xf32>
    %4 = tensor.empty(%dim_1) : tensor<?x4xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?xi32>, tensor<4xf32>) outs(%4 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg1[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %c0_i32_2 = arith.constant 0 : i32
    %c0_3 = arith.constant 0 : index
    %dim_4 = tensor.dim %arg2, %c0_3 : tensor<?x2xi32>
    %6 = tensor.empty(%dim_4) : tensor<?xi32>
    %7 = linalg.fill ins(%c0_i32_2 : i32) outs(%6 : tensor<?xi32>) -> tensor<?xi32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%7 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_5 = arith.constant 0 : index
    %dim_6 = tensor.dim %8, %c0_5 : tensor<?xi32>
    %9 = tensor.empty() : tensor<4xf32>
    %10 = tensor.empty(%dim_6) : tensor<?x4xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %9 : tensor<?xi32>, tensor<4xf32>) outs(%10 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg3[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %c0_i32_7 = arith.constant 0 : i32
    %c0_8 = arith.constant 0 : index
    %dim_9 = tensor.dim %arg4, %c0_8 : tensor<?x2xi32>
    %12 = tensor.empty(%dim_9) : tensor<?xi32>
    %13 = linalg.fill ins(%c0_i32_7 : i32) outs(%12 : tensor<?xi32>) -> tensor<?xi32>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%13 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_10 = arith.constant 0 : index
    %dim_11 = tensor.dim %14, %c0_10 : tensor<?xi32>
    %15 = tensor.empty() : tensor<4xf32>
    %16 = tensor.empty(%dim_11) : tensor<?x4xf32>
    %17 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<?xi32>, tensor<4xf32>) outs(%16 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg5[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    return %5, %11, %17 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump After ConvertToSignlessPass (convert-to-signless) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %cst = arith.constant dense<0> : tensor<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_0 = arith.constant 0 : index
    %dim_1 = tensor.dim %2, %c0_0 : tensor<?xi32>
    %3 = tensor.empty() : tensor<4xf32>
    %4 = tensor.empty(%dim_1) : tensor<?x4xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?xi32>, tensor<4xf32>) outs(%4 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg1[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %c0_i32_2 = arith.constant 0 : i32
    %c0_3 = arith.constant 0 : index
    %dim_4 = tensor.dim %arg2, %c0_3 : tensor<?x2xi32>
    %6 = tensor.empty(%dim_4) : tensor<?xi32>
    %7 = linalg.fill ins(%c0_i32_2 : i32) outs(%6 : tensor<?xi32>) -> tensor<?xi32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%7 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_5 = arith.constant 0 : index
    %dim_6 = tensor.dim %8, %c0_5 : tensor<?xi32>
    %9 = tensor.empty() : tensor<4xf32>
    %10 = tensor.empty(%dim_6) : tensor<?x4xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %9 : tensor<?xi32>, tensor<4xf32>) outs(%10 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg3[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %c0_i32_7 = arith.constant 0 : i32
    %c0_8 = arith.constant 0 : index
    %dim_9 = tensor.dim %arg4, %c0_8 : tensor<?x2xi32>
    %12 = tensor.empty(%dim_9) : tensor<?xi32>
    %13 = linalg.fill ins(%c0_i32_7 : i32) outs(%12 : tensor<?xi32>) -> tensor<?xi32>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%13 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_10 = arith.constant 0 : index
    %dim_11 = tensor.dim %14, %c0_10 : tensor<?xi32>
    %15 = tensor.empty() : tensor<4xf32>
    %16 = tensor.empty(%dim_11) : tensor<?x4xf32>
    %17 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<?xi32>, tensor<4xf32>) outs(%16 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg5[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    return %5, %11, %17 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %cst = arith.constant dense<0> : tensor<i32>
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_0 = arith.constant 0 : index
    %dim_1 = tensor.dim %2, %c0_0 : tensor<?xi32>
    %3 = tensor.empty() : tensor<4xf32>
    %4 = tensor.empty(%dim_1) : tensor<?x4xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<?xi32>, tensor<4xf32>) outs(%4 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg1[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %c0_i32_2 = arith.constant 0 : i32
    %c0_3 = arith.constant 0 : index
    %dim_4 = tensor.dim %arg2, %c0_3 : tensor<?x2xi32>
    %6 = tensor.empty(%dim_4) : tensor<?xi32>
    %7 = linalg.fill ins(%c0_i32_2 : i32) outs(%6 : tensor<?xi32>) -> tensor<?xi32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%7 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_5 = arith.constant 0 : index
    %dim_6 = tensor.dim %8, %c0_5 : tensor<?xi32>
    %9 = tensor.empty() : tensor<4xf32>
    %10 = tensor.empty(%dim_6) : tensor<?x4xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %9 : tensor<?xi32>, tensor<4xf32>) outs(%10 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg3[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %c0_i32_7 = arith.constant 0 : i32
    %c0_8 = arith.constant 0 : index
    %dim_9 = tensor.dim %arg4, %c0_8 : tensor<?x2xi32>
    %12 = tensor.empty(%dim_9) : tensor<?xi32>
    %13 = linalg.fill ins(%c0_i32_7 : i32) outs(%12 : tensor<?xi32>) -> tensor<?xi32>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%13 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %from_elements = tensor.from_elements %out : tensor<i32>
      %from_elements_12 = tensor.from_elements %in : tensor<i32>
      %extracted = tensor.extract %from_elements[] : tensor<i32>
      %extracted_13 = tensor.extract %from_elements_12[] : tensor<i32>
      %18 = arith.addi %extracted, %extracted_13 : i32
      %from_elements_14 = tensor.from_elements %18 : tensor<i32>
      %extracted_15 = tensor.extract %from_elements_14[] : tensor<i32>
      linalg.yield %extracted_15 : i32
    } -> tensor<?xi32>
    %c0_10 = arith.constant 0 : index
    %dim_11 = tensor.dim %14, %c0_10 : tensor<?xi32>
    %15 = tensor.empty() : tensor<4xf32>
    %16 = tensor.empty(%dim_11) : tensor<?x4xf32>
    %17 = linalg.generic {indexing_maps = [#map1, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %15 : tensor<?xi32>, tensor<4xf32>) outs(%16 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %in_12: f32, %out: f32):
      %18 = arith.index_cast %in : i32 to index
      %19 = linalg.index 1 : index
      %extracted = tensor.extract %arg5[%18, %19] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    return %5, %11, %17 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_0 = tensor.dim %2, %c0 : tensor<?xi32>
    %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
    %5 = tensor.empty(%dim_1) : tensor<?xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_2 = tensor.dim %7, %c0 : tensor<?xi32>
    %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
    %10 = tensor.empty(%dim_3) : tensor<?xi32>
    %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_4 = tensor.dim %12, %c0 : tensor<?xi32>
    %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump Before CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_0 = tensor.dim %2, %c0 : tensor<?xi32>
  %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %5 = tensor.empty(%dim_1) : tensor<?xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_2 = tensor.dim %7, %c0 : tensor<?xi32>
  %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %10 = tensor.empty(%dim_3) : tensor<?xi32>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_4 = tensor.dim %12, %c0 : tensor<?xi32>
  %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_0 = tensor.dim %2, %c0 : tensor<?xi32>
  %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %5 = tensor.empty(%dim_1) : tensor<?xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_2 = tensor.dim %7, %c0 : tensor<?xi32>
  %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %10 = tensor.empty(%dim_3) : tensor<?xi32>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_4 = tensor.dim %12, %c0 : tensor<?xi32>
  %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before ConvertComplexToStandard (convert-complex-to-standard) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_0 = tensor.dim %2, %c0 : tensor<?xi32>
  %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %5 = tensor.empty(%dim_1) : tensor<?xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_2 = tensor.dim %7, %c0 : tensor<?xi32>
  %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %10 = tensor.empty(%dim_3) : tensor<?xi32>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_4 = tensor.dim %12, %c0 : tensor<?xi32>
  %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After ConvertComplexToStandard (convert-complex-to-standard) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_0 = tensor.dim %2, %c0 : tensor<?xi32>
  %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %5 = tensor.empty(%dim_1) : tensor<?xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_2 = tensor.dim %7, %c0 : tensor<?xi32>
  %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %10 = tensor.empty(%dim_3) : tensor<?xi32>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_4 = tensor.dim %12, %c0 : tensor<?xi32>
  %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before ResolveRankedShapeTypeResultDims (resolve-ranked-shaped-type-result-dims) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_0 = tensor.dim %2, %c0 : tensor<?xi32>
    %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
    %5 = tensor.empty(%dim_1) : tensor<?xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_2 = tensor.dim %7, %c0 : tensor<?xi32>
    %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
    %10 = tensor.empty(%dim_3) : tensor<?xi32>
    %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_4 = tensor.dim %12, %c0 : tensor<?xi32>
    %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump After ResolveRankedShapeTypeResultDims (resolve-ranked-shaped-type-result-dims) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_0 = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
    %5 = tensor.empty(%dim_1) : tensor<?xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_2 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
    %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
    %10 = tensor.empty(%dim_3) : tensor<?xi32>
    %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_4 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
    %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_0 = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
    %5 = tensor.empty(%dim_1) : tensor<?xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_2 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
    %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
    %10 = tensor.empty(%dim_3) : tensor<?xi32>
    %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_4 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
    %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_0 = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
    %5 = tensor.empty(%dim_1) : tensor<?xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_2 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
    %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
    %10 = tensor.empty(%dim_3) : tensor<?xi32>
    %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_4 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
    %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump Before CustomLinalgElementwiseOpFusion (custom-linalg-cwise-fusion) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_0 = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %5 = tensor.empty(%dim_1) : tensor<?xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_2 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %10 = tensor.empty(%dim_3) : tensor<?xi32>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_4 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After CustomLinalgElementwiseOpFusion (custom-linalg-cwise-fusion) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_0 = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %5 = tensor.empty(%dim_1) : tensor<?xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_2 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %10 = tensor.empty(%dim_3) : tensor<?xi32>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_4 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before EmptyTensorToAllocTensor (empty-tensor-to-alloc-tensor) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = tensor.empty(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_0 = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %3 = tensor.empty(%dim_0) : tensor<?x4xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %5 = tensor.empty(%dim_1) : tensor<?xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_2 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %8 = tensor.empty(%dim_2) : tensor<?x4xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %10 = tensor.empty(%dim_3) : tensor<?xi32>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_4 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %13 = tensor.empty(%dim_4) : tensor<?x4xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump After EmptyTensorToAllocTensor (empty-tensor-to-alloc-tensor) //----- //
func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %0 = bufferization.alloc_tensor(%dim) : tensor<?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_0 = tensor.dim %arg0, %c0 : tensor<?x2xi32>
  %3 = bufferization.alloc_tensor(%dim_0) : tensor<?x4xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %5 = bufferization.alloc_tensor(%dim_1) : tensor<?xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_2 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
  %8 = bufferization.alloc_tensor(%dim_2) : tensor<?x4xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %10 = bufferization.alloc_tensor(%dim_3) : tensor<?xi32>
  %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %15 = arith.addi %out, %in : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>
  %dim_4 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
  %13 = bufferization.alloc_tensor(%dim_4) : tensor<?x4xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %15 = arith.index_cast %in : i32 to index
    %16 = linalg.index 1 : index
    %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x4xf32>
  return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
}

// -----// IR Dump Before CustomComputeOpAndFuncBufferizePass (custom-computeop-and-func-bufferize) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @cuda_kernel_0(%arg0: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: tensor<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %0 = bufferization.alloc_tensor(%dim) : tensor<?xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?xi32>) -> tensor<?xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x2xi32>) outs(%1 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_0 = tensor.dim %arg0, %c0 : tensor<?x2xi32>
    %3 = bufferization.alloc_tensor(%dim_0) : tensor<?x4xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<?xi32>) outs(%3 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg1[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_1 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
    %5 = bufferization.alloc_tensor(%dim_1) : tensor<?xi32>
    %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<?xi32>) -> tensor<?xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x2xi32>) outs(%6 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_2 = tensor.dim %arg2, %c0 : tensor<?x2xi32>
    %8 = bufferization.alloc_tensor(%dim_2) : tensor<?x4xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<?xi32>) outs(%8 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg3[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    %dim_3 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
    %10 = bufferization.alloc_tensor(%dim_3) : tensor<?xi32>
    %11 = linalg.fill ins(%c0_i32 : i32) outs(%10 : tensor<?xi32>) -> tensor<?xi32>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg4 : tensor<?x2xi32>) outs(%11 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %15 = arith.addi %out, %in : i32
      linalg.yield %15 : i32
    } -> tensor<?xi32>
    %dim_4 = tensor.dim %arg4, %c0 : tensor<?x2xi32>
    %13 = bufferization.alloc_tensor(%dim_4) : tensor<?x4xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<?xi32>) outs(%13 : tensor<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %15 = arith.index_cast %in : i32 to index
      %16 = linalg.index 1 : index
      %extracted = tensor.extract %arg5[%15, %16] : tensor<?x4xf32>
      linalg.yield %extracted : f32
    } -> tensor<?x4xf32>
    return %4, %9, %14 : tensor<?x4xf32>, tensor<?x4xf32>, tensor<?x4xf32>
  }
}


// -----// IR Dump After CustomComputeOpAndFuncBufferizePass (custom-computeop-and-func-bufferize) //----- //
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
    %0 = bufferization.to_tensor %arg5 : memref<?x4xf32>
    %1 = bufferization.to_tensor %arg4 : memref<?x2xi32>
    %2 = bufferization.to_tensor %arg3 : memref<?x4xf32>
    %3 = bufferization.to_tensor %arg2 : memref<?x2xi32>
    %4 = bufferization.to_tensor %arg1 : memref<?x4xf32>
    %5 = bufferization.to_tensor %arg0 : memref<?x2xi32>
    %6 = bufferization.to_memref %0 : memref<?x4xf32>
    %7 = bufferization.to_memref %1 : memref<?x2xi32>
    %8 = bufferization.to_memref %1 : memref<?x2xi32>
    %9 = bufferization.to_memref %1 : memref<?x2xi32>
    %10 = bufferization.to_memref %2 : memref<?x4xf32>
    %11 = bufferization.to_memref %3 : memref<?x2xi32>
    %12 = bufferization.to_memref %3 : memref<?x2xi32>
    %13 = bufferization.to_memref %3 : memref<?x2xi32>
    %14 = bufferization.to_memref %4 : memref<?x4xf32>
    %15 = bufferization.to_memref %5 : memref<?x2xi32>
    %16 = bufferization.to_memref %5 : memref<?x2xi32>
    %17 = bufferization.to_memref %5 : memref<?x2xi32>
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %dim = memref.dim %17, %c0 : memref<?x2xi32>
    %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
    %18 = bufferization.to_tensor %alloc : memref<?xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<?xi32>)
    %19 = bufferization.to_tensor %alloc : memref<?xi32>
    %c0_0 = arith.constant 0 : index
    %dim_1 = memref.dim %alloc, %c0_0 : memref<?xi32>
    %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
    memref.copy %alloc, %alloc_2 : memref<?xi32> to memref<?xi32>
    %20 = bufferization.to_tensor %alloc_2 : memref<?xi32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%16 : memref<?x2xi32>) outs(%alloc_2 : memref<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %39 = arith.addi %out, %in : i32
      linalg.yield %39 : i32
    }
    %21 = bufferization.to_tensor %alloc_2 : memref<?xi32>
    %dim_3 = memref.dim %15, %c0 : memref<?x2xi32>
    %c4 = arith.constant 4 : index
    %alloc_4 = memref.alloc(%dim_3) {alignment = 128 : i64} : memref<?x4xf32>
    %22 = bufferization.to_tensor %alloc_4 : memref<?x4xf32>
    linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_2 : memref<?xi32>) outs(%alloc_4 : memref<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %39 = arith.index_cast %in : i32 to index
      %40 = linalg.index 1 : index
      %41 = memref.load %14[%39, %40] : memref<?x4xf32>
      linalg.yield %41 : f32
    }
    %23 = bufferization.to_tensor %alloc_4 : memref<?x4xf32>
    %24 = bufferization.to_memref %23 : memref<?x4xf32>
    %dim_5 = memref.dim %13, %c0 : memref<?x2xi32>
    %alloc_6 = memref.alloc(%dim_5) {alignment = 128 : i64} : memref<?xi32>
    %25 = bufferization.to_tensor %alloc_6 : memref<?xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_6 : memref<?xi32>)
    %26 = bufferization.to_tensor %alloc_6 : memref<?xi32>
    %c0_7 = arith.constant 0 : index
    %dim_8 = memref.dim %alloc_6, %c0_7 : memref<?xi32>
    %alloc_9 = memref.alloc(%dim_8) {alignment = 128 : i64} : memref<?xi32>
    memref.copy %alloc_6, %alloc_9 : memref<?xi32> to memref<?xi32>
    %27 = bufferization.to_tensor %alloc_9 : memref<?xi32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%12 : memref<?x2xi32>) outs(%alloc_9 : memref<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %39 = arith.addi %out, %in : i32
      linalg.yield %39 : i32
    }
    %28 = bufferization.to_tensor %alloc_9 : memref<?xi32>
    %dim_10 = memref.dim %11, %c0 : memref<?x2xi32>
    %c4_11 = arith.constant 4 : index
    %alloc_12 = memref.alloc(%dim_10) {alignment = 128 : i64} : memref<?x4xf32>
    %29 = bufferization.to_tensor %alloc_12 : memref<?x4xf32>
    linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_9 : memref<?xi32>) outs(%alloc_12 : memref<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %39 = arith.index_cast %in : i32 to index
      %40 = linalg.index 1 : index
      %41 = memref.load %10[%39, %40] : memref<?x4xf32>
      linalg.yield %41 : f32
    }
    %30 = bufferization.to_tensor %alloc_12 : memref<?x4xf32>
    %31 = bufferization.to_memref %30 : memref<?x4xf32>
    %dim_13 = memref.dim %9, %c0 : memref<?x2xi32>
    %alloc_14 = memref.alloc(%dim_13) {alignment = 128 : i64} : memref<?xi32>
    %32 = bufferization.to_tensor %alloc_14 : memref<?xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc_14 : memref<?xi32>)
    %33 = bufferization.to_tensor %alloc_14 : memref<?xi32>
    %c0_15 = arith.constant 0 : index
    %dim_16 = memref.dim %alloc_14, %c0_15 : memref<?xi32>
    %alloc_17 = memref.alloc(%dim_16) {alignment = 128 : i64} : memref<?xi32>
    memref.copy %alloc_14, %alloc_17 : memref<?xi32> to memref<?xi32>
    %34 = bufferization.to_tensor %alloc_17 : memref<?xi32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%8 : memref<?x2xi32>) outs(%alloc_17 : memref<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %39 = arith.addi %out, %in : i32
      linalg.yield %39 : i32
    }
    %35 = bufferization.to_tensor %alloc_17 : memref<?xi32>
    %dim_18 = memref.dim %7, %c0 : memref<?x2xi32>
    %c4_19 = arith.constant 4 : index
    %alloc_20 = memref.alloc(%dim_18) {alignment = 128 : i64} : memref<?x4xf32>
    %36 = bufferization.to_tensor %alloc_20 : memref<?x4xf32>
    linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_17 : memref<?xi32>) outs(%alloc_20 : memref<?x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %39 = arith.index_cast %in : i32 to index
      %40 = linalg.index 1 : index
      %41 = memref.load %6[%39, %40] : memref<?x4xf32>
      linalg.yield %41 : f32
    }
    %37 = bufferization.to_tensor %alloc_20 : memref<?x4xf32>
    %38 = bufferization.to_memref %37 : memref<?x4xf32>
    return %24, %31, %38 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %0 = bufferization.to_tensor %arg5 : memref<?x4xf32>
  %1 = bufferization.to_tensor %arg4 : memref<?x2xi32>
  %2 = bufferization.to_tensor %arg3 : memref<?x4xf32>
  %3 = bufferization.to_tensor %arg2 : memref<?x2xi32>
  %4 = bufferization.to_tensor %arg1 : memref<?x4xf32>
  %5 = bufferization.to_tensor %arg0 : memref<?x2xi32>
  %6 = bufferization.to_memref %0 : memref<?x4xf32>
  %7 = bufferization.to_memref %1 : memref<?x2xi32>
  %8 = bufferization.to_memref %1 : memref<?x2xi32>
  %9 = bufferization.to_memref %1 : memref<?x2xi32>
  %10 = bufferization.to_memref %2 : memref<?x4xf32>
  %11 = bufferization.to_memref %3 : memref<?x2xi32>
  %12 = bufferization.to_memref %3 : memref<?x2xi32>
  %13 = bufferization.to_memref %3 : memref<?x2xi32>
  %14 = bufferization.to_memref %4 : memref<?x4xf32>
  %15 = bufferization.to_memref %5 : memref<?x2xi32>
  %16 = bufferization.to_memref %5 : memref<?x2xi32>
  %17 = bufferization.to_memref %5 : memref<?x2xi32>
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %17, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  %18 = bufferization.to_tensor %alloc : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<?xi32>)
  %19 = bufferization.to_tensor %alloc : memref<?xi32>
  %c0_0 = arith.constant 0 : index
  %dim_1 = memref.dim %alloc, %c0_0 : memref<?xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc, %alloc_2 : memref<?xi32> to memref<?xi32>
  %20 = bufferization.to_tensor %alloc_2 : memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%16 : memref<?x2xi32>) outs(%alloc_2 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %39 = arith.addi %out, %in : i32
    linalg.yield %39 : i32
  }
  %21 = bufferization.to_tensor %alloc_2 : memref<?xi32>
  %dim_3 = memref.dim %15, %c0 : memref<?x2xi32>
  %c4 = arith.constant 4 : index
  %alloc_4 = memref.alloc(%dim_3) {alignment = 128 : i64} : memref<?x4xf32>
  %22 = bufferization.to_tensor %alloc_4 : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_2 : memref<?xi32>) outs(%alloc_4 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %39 = arith.index_cast %in : i32 to index
    %40 = linalg.index 1 : index
    %41 = memref.load %14[%39, %40] : memref<?x4xf32>
    linalg.yield %41 : f32
  }
  %23 = bufferization.to_tensor %alloc_4 : memref<?x4xf32>
  %24 = bufferization.to_memref %23 : memref<?x4xf32>
  %dim_5 = memref.dim %13, %c0 : memref<?x2xi32>
  %alloc_6 = memref.alloc(%dim_5) {alignment = 128 : i64} : memref<?xi32>
  %25 = bufferization.to_tensor %alloc_6 : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_6 : memref<?xi32>)
  %26 = bufferization.to_tensor %alloc_6 : memref<?xi32>
  %c0_7 = arith.constant 0 : index
  %dim_8 = memref.dim %alloc_6, %c0_7 : memref<?xi32>
  %alloc_9 = memref.alloc(%dim_8) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc_6, %alloc_9 : memref<?xi32> to memref<?xi32>
  %27 = bufferization.to_tensor %alloc_9 : memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%12 : memref<?x2xi32>) outs(%alloc_9 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %39 = arith.addi %out, %in : i32
    linalg.yield %39 : i32
  }
  %28 = bufferization.to_tensor %alloc_9 : memref<?xi32>
  %dim_10 = memref.dim %11, %c0 : memref<?x2xi32>
  %c4_11 = arith.constant 4 : index
  %alloc_12 = memref.alloc(%dim_10) {alignment = 128 : i64} : memref<?x4xf32>
  %29 = bufferization.to_tensor %alloc_12 : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_9 : memref<?xi32>) outs(%alloc_12 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %39 = arith.index_cast %in : i32 to index
    %40 = linalg.index 1 : index
    %41 = memref.load %10[%39, %40] : memref<?x4xf32>
    linalg.yield %41 : f32
  }
  %30 = bufferization.to_tensor %alloc_12 : memref<?x4xf32>
  %31 = bufferization.to_memref %30 : memref<?x4xf32>
  %dim_13 = memref.dim %9, %c0 : memref<?x2xi32>
  %alloc_14 = memref.alloc(%dim_13) {alignment = 128 : i64} : memref<?xi32>
  %32 = bufferization.to_tensor %alloc_14 : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_14 : memref<?xi32>)
  %33 = bufferization.to_tensor %alloc_14 : memref<?xi32>
  %c0_15 = arith.constant 0 : index
  %dim_16 = memref.dim %alloc_14, %c0_15 : memref<?xi32>
  %alloc_17 = memref.alloc(%dim_16) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc_14, %alloc_17 : memref<?xi32> to memref<?xi32>
  %34 = bufferization.to_tensor %alloc_17 : memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%8 : memref<?x2xi32>) outs(%alloc_17 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %39 = arith.addi %out, %in : i32
    linalg.yield %39 : i32
  }
  %35 = bufferization.to_tensor %alloc_17 : memref<?xi32>
  %dim_18 = memref.dim %7, %c0 : memref<?x2xi32>
  %c4_19 = arith.constant 4 : index
  %alloc_20 = memref.alloc(%dim_18) {alignment = 128 : i64} : memref<?x4xf32>
  %36 = bufferization.to_tensor %alloc_20 : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_17 : memref<?xi32>) outs(%alloc_20 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %39 = arith.index_cast %in : i32 to index
    %40 = linalg.index 1 : index
    %41 = memref.load %6[%39, %40] : memref<?x4xf32>
    linalg.yield %41 : f32
  }
  %37 = bufferization.to_tensor %alloc_20 : memref<?x4xf32>
  %38 = bufferization.to_memref %37 : memref<?x4xf32>
  return %24, %31, %38 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<?xi32>)
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc, %alloc_0 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<?x2xi32>) outs(%alloc_0 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %dim_1 = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_0 : memref<?xi32>) outs(%alloc_2 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg1[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  %dim_3 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_4 = memref.alloc(%dim_3) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_4 : memref<?xi32>)
  %alloc_5 = memref.alloc(%dim_3) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc_4, %alloc_5 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<?x2xi32>) outs(%alloc_5 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %dim_6 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_7 = memref.alloc(%dim_6) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_5 : memref<?xi32>) outs(%alloc_7 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg3[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  %dim_8 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_9 = memref.alloc(%dim_8) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_9 : memref<?xi32>)
  %alloc_10 = memref.alloc(%dim_8) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc_9, %alloc_10 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : memref<?x2xi32>) outs(%alloc_10 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %dim_11 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_12 = memref.alloc(%dim_11) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_10 : memref<?xi32>) outs(%alloc_12 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg5[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  return %alloc_2, %alloc_7, %alloc_12 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<?xi32>)
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc, %alloc_0 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<?x2xi32>) outs(%alloc_0 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %dim_1 = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_0 : memref<?xi32>) outs(%alloc_2 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg1[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  %dim_3 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_4 = memref.alloc(%dim_3) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_4 : memref<?xi32>)
  %alloc_5 = memref.alloc(%dim_3) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc_4, %alloc_5 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<?x2xi32>) outs(%alloc_5 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %dim_6 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_7 = memref.alloc(%dim_6) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_5 : memref<?xi32>) outs(%alloc_7 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg3[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  %dim_8 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_9 = memref.alloc(%dim_8) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_9 : memref<?xi32>)
  %alloc_10 = memref.alloc(%dim_8) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc_9, %alloc_10 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : memref<?x2xi32>) outs(%alloc_10 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %dim_11 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_12 = memref.alloc(%dim_11) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_10 : memref<?xi32>) outs(%alloc_12 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg5[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  return %alloc_2, %alloc_7, %alloc_12 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<?xi32>)
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc, %alloc_0 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<?x2xi32>) outs(%alloc_0 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %alloc_1 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_0 : memref<?xi32>) outs(%alloc_1 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg1[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  %dim_2 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_3 = memref.alloc(%dim_2) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_3 : memref<?xi32>)
  %alloc_4 = memref.alloc(%dim_2) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc_3, %alloc_4 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<?x2xi32>) outs(%alloc_4 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %alloc_5 = memref.alloc(%dim_2) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_4 : memref<?xi32>) outs(%alloc_5 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg3[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  %dim_6 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_7 = memref.alloc(%dim_6) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_7 : memref<?xi32>)
  %alloc_8 = memref.alloc(%dim_6) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc_7, %alloc_8 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : memref<?x2xi32>) outs(%alloc_8 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %alloc_9 = memref.alloc(%dim_6) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_8 : memref<?xi32>) outs(%alloc_9 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg5[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  return %alloc_1, %alloc_5, %alloc_9 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before CopyCleanupPass (custom-copy-cleanup) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<?xi32>)
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc, %alloc_0 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<?x2xi32>) outs(%alloc_0 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %alloc_1 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_0 : memref<?xi32>) outs(%alloc_1 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg1[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  %dim_2 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_3 = memref.alloc(%dim_2) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_3 : memref<?xi32>)
  %alloc_4 = memref.alloc(%dim_2) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc_3, %alloc_4 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg2 : memref<?x2xi32>) outs(%alloc_4 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %alloc_5 = memref.alloc(%dim_2) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_4 : memref<?xi32>) outs(%alloc_5 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg3[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  %dim_6 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_7 = memref.alloc(%dim_6) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc_7 : memref<?xi32>)
  %alloc_8 = memref.alloc(%dim_6) {alignment = 128 : i64} : memref<?xi32>
  memref.copy %alloc_7, %alloc_8 : memref<?xi32> to memref<?xi32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg4 : memref<?x2xi32>) outs(%alloc_8 : memref<?xi32>) {
  ^bb0(%in: i32, %out: i32):
    %0 = arith.addi %out, %in : i32
    linalg.yield %0 : i32
  }
  %alloc_9 = memref.alloc(%dim_6) {alignment = 128 : i64} : memref<?x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_8 : memref<?xi32>) outs(%alloc_9 : memref<?x4xf32>) {
  ^bb0(%in: i32, %out: f32):
    %0 = arith.index_cast %in : i32 to index
    %1 = linalg.index 1 : index
    %2 = memref.load %arg5[%0, %1] : memref<?x4xf32>
    linalg.yield %2 : f32
  }
  return %alloc_1, %alloc_5, %alloc_9 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After CopyCleanupPass (custom-copy-cleanup) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<?xi32>)
  memref.copy %alloc, %alloc : memref<?xi32> to memref<?xi32>
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
  memref.copy %alloc_2, %alloc_2 : memref<?xi32> to memref<?xi32>
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
  memref.copy %alloc_5, %alloc_5 : memref<?xi32> to memref<?xi32>
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

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<?xi32>)
  memref.copy %alloc, %alloc : memref<?xi32> to memref<?xi32>
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
  memref.copy %alloc_2, %alloc_2 : memref<?xi32> to memref<?xi32>
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
  memref.copy %alloc_5, %alloc_5 : memref<?xi32> to memref<?xi32>
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

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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

// -----// IR Dump Before CSE (cse) //----- //
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

// -----// IR Dump After CSE (cse) //----- //
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

// -----// IR Dump Before CopyImplPass (copy-impl-pass) //----- //
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

// -----// IR Dump After CopyImplPass (copy-impl-pass) //----- //
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

// -----// IR Dump Before FloatCompute (float-compute) //----- //
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

// -----// IR Dump After FloatCompute (float-compute) //----- //
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

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
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

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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

// -----// IR Dump Before CSE (cse) //----- //
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

// -----// IR Dump After CSE (cse) //----- //
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

// -----// IR Dump Before LegalizeTanhToApproximationPass (mhlo-legalize-trigonometric-to-approximation) //----- //
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

// -----// IR Dump After LegalizeTanhToApproximationPass (mhlo-legalize-trigonometric-to-approximation) //----- //
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

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
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

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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

// -----// IR Dump Before CSE (cse) //----- //
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

// -----// IR Dump After CSE (cse) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before AFReduceFusionInitAndEpi (af-reduce-fusion-init-and-epi) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After AFReduceFusionInitAndEpi (af-reduce-fusion-init-and-epi) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before AFReduceConvert (af-reduce-convert) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After AFReduceConvert (af-reduce-convert) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before AFReduceSpecOpt (af-reduce-spec-opt) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After AFReduceSpecOpt (af-reduce-spec-opt) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before AFReduceWarpTile (af-reduce-warp-tile) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After AFReduceWarpTile (af-reduce-warp-tile) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before CollapseParallelLoopsTo1DPass (collapse-parallel-loops-to-1d) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_0[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_1, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_2[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg3[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_3[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %0 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %1 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %2 = arith.addi %1, %0 : i32
      memref.store %2, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%dim_4, %c4) step (%c1, %c1) {
    %0 = memref.load %alloc_5[%arg6] : memref<?xi32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg5[%1, %arg7] : memref<?x4xf32>
    memref.store %2, %alloc_6[%arg6, %arg7] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After CollapseParallelLoopsTo1DPass (collapse-parallel-loops-to-1d) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %6 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %7 = memref.load %alloc[%arg6] : memref<?xi32>
      %8 = arith.addi %7, %6 : i32
      memref.store %8, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %c0_1 = arith.constant 0 : index
  %c1_2 = arith.constant 1 : index
  %c1_3 = arith.constant 1 : index
  %0 = arith.muli %c1_3, %dim : index
  %1 = arith.muli %0, %c4 : index
  scf.parallel (%arg6) = (%c0_1) to (%1) step (%c1_2) {
    %6 = arith.remsi %arg6, %c4 : index
    %7 = arith.divsi %arg6, %c4 : index
    %8 = memref.load %alloc[%7] : memref<?xi32>
    %9 = arith.index_cast %8 : i32 to index
    %10 = memref.load %arg1[%9, %6] : memref<?x4xf32>
    memref.store %10, %alloc_0[%7, %6] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %6 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %7 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %8 = arith.addi %7, %6 : i32
      memref.store %8, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %c0_7 = arith.constant 0 : index
  %c1_8 = arith.constant 1 : index
  %c1_9 = arith.constant 1 : index
  %2 = arith.muli %c1_9, %dim_4 : index
  %3 = arith.muli %2, %c4 : index
  scf.parallel (%arg6) = (%c0_7) to (%3) step (%c1_8) {
    %6 = arith.remsi %arg6, %c4 : index
    %7 = arith.divsi %arg6, %c4 : index
    %8 = memref.load %alloc_5[%7] : memref<?xi32>
    %9 = arith.index_cast %8 : i32 to index
    %10 = memref.load %arg3[%9, %6] : memref<?x4xf32>
    memref.store %10, %alloc_6[%7, %6] : memref<?x4xf32>
    scf.yield
  }
  %dim_10 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_11 = memref.alloc(%dim_10) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_10) step (%c1) {
    memref.store %c0_i32, %alloc_11[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_10) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %6 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %7 = memref.load %alloc_11[%arg6] : memref<?xi32>
      %8 = arith.addi %7, %6 : i32
      memref.store %8, %alloc_11[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_12 = memref.alloc(%dim_10) {alignment = 128 : i64} : memref<?x4xf32>
  %c0_13 = arith.constant 0 : index
  %c1_14 = arith.constant 1 : index
  %c1_15 = arith.constant 1 : index
  %4 = arith.muli %c1_15, %dim_10 : index
  %5 = arith.muli %4, %c4 : index
  scf.parallel (%arg6) = (%c0_13) to (%5) step (%c1_14) {
    %6 = arith.remsi %arg6, %c4 : index
    %7 = arith.divsi %arg6, %c4 : index
    %8 = memref.load %alloc_11[%7] : memref<?xi32>
    %9 = arith.index_cast %8 : i32 to index
    %10 = memref.load %arg5[%9, %6] : memref<?x4xf32>
    memref.store %10, %alloc_12[%7, %6] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_6, %alloc_12 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %6 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %7 = memref.load %alloc[%arg6] : memref<?xi32>
      %8 = arith.addi %7, %6 : i32
      memref.store %8, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %c0_1 = arith.constant 0 : index
  %c1_2 = arith.constant 1 : index
  %c1_3 = arith.constant 1 : index
  %0 = arith.muli %c1_3, %dim : index
  %1 = arith.muli %0, %c4 : index
  scf.parallel (%arg6) = (%c0_1) to (%1) step (%c1_2) {
    %6 = arith.remsi %arg6, %c4 : index
    %7 = arith.divsi %arg6, %c4 : index
    %8 = memref.load %alloc[%7] : memref<?xi32>
    %9 = arith.index_cast %8 : i32 to index
    %10 = memref.load %arg1[%9, %6] : memref<?x4xf32>
    memref.store %10, %alloc_0[%7, %6] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %6 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %7 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %8 = arith.addi %7, %6 : i32
      memref.store %8, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %c0_7 = arith.constant 0 : index
  %c1_8 = arith.constant 1 : index
  %c1_9 = arith.constant 1 : index
  %2 = arith.muli %c1_9, %dim_4 : index
  %3 = arith.muli %2, %c4 : index
  scf.parallel (%arg6) = (%c0_7) to (%3) step (%c1_8) {
    %6 = arith.remsi %arg6, %c4 : index
    %7 = arith.divsi %arg6, %c4 : index
    %8 = memref.load %alloc_5[%7] : memref<?xi32>
    %9 = arith.index_cast %8 : i32 to index
    %10 = memref.load %arg3[%9, %6] : memref<?x4xf32>
    memref.store %10, %alloc_6[%7, %6] : memref<?x4xf32>
    scf.yield
  }
  %dim_10 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_11 = memref.alloc(%dim_10) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_10) step (%c1) {
    memref.store %c0_i32, %alloc_11[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_10) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %6 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %7 = memref.load %alloc_11[%arg6] : memref<?xi32>
      %8 = arith.addi %7, %6 : i32
      memref.store %8, %alloc_11[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_12 = memref.alloc(%dim_10) {alignment = 128 : i64} : memref<?x4xf32>
  %c0_13 = arith.constant 0 : index
  %c1_14 = arith.constant 1 : index
  %c1_15 = arith.constant 1 : index
  %4 = arith.muli %c1_15, %dim_10 : index
  %5 = arith.muli %4, %c4 : index
  scf.parallel (%arg6) = (%c0_13) to (%5) step (%c1_14) {
    %6 = arith.remsi %arg6, %c4 : index
    %7 = arith.divsi %arg6, %c4 : index
    %8 = memref.load %alloc_11[%7] : memref<?xi32>
    %9 = arith.index_cast %8 : i32 to index
    %10 = memref.load %arg5[%9, %6] : memref<?x4xf32>
    memref.store %10, %alloc_12[%7, %6] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_6, %alloc_12 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before PromoteBuffersToStack (promote-buffers-to-stack) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After PromoteBuffersToStack (promote-buffers-to-stack) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before AFMarkHostParallelOp (af-mark-host-parallel-op) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After AFMarkHostParallelOp (af-mark-host-parallel-op) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before TileLoopsPass (tile-loops) //----- //
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
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg0[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg1[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_0[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    memref.store %c0_i32, %alloc_2[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg2[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_2[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_2[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_2[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg3[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_3[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    memref.store %c0_i32, %alloc_5[%arg6] : memref<?xi32>
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c1) {
    scf.for %arg7 = %c0 to %c2 step %c1 {
      %3 = memref.load %arg4[%arg6, %arg7] : memref<?x2xi32>
      %4 = memref.load %alloc_5[%arg6] : memref<?xi32>
      %5 = arith.addi %4, %3 : i32
      memref.store %5, %alloc_5[%arg6] : memref<?xi32>
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c1) {
    %3 = arith.remsi %arg6, %c4 : index
    %4 = arith.divsi %arg6, %c4 : index
    %5 = memref.load %alloc_5[%4] : memref<?xi32>
    %6 = arith.index_cast %5 : i32 to index
    %7 = memref.load %arg5[%6, %3] : memref<?x4xf32>
    memref.store %7, %alloc_6[%4, %3] : memref<?x4xf32>
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After TileLoopsPass (tile-loops) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg0[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %0, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg1[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_0[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim_1, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim_1, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg2[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_2[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_2[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %1, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_2[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg3[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_3[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim_4, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim_4, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg4[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_5[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_5[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %2, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_5[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg5[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_6[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg0[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %0, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg1[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_0[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim_1, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim_1, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg2[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_2[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_2[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %1, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_2[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg3[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_3[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim_4, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %dim_4, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg4[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_5[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_5[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c512) {
    %3 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %2, %arg6)
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_5[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg5[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_6[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg0[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg1[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_0[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg2[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_2[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_2[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_2[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg3[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_3[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg4[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_5[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_5[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%2]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_5[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg5[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_6[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg0[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg1[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_0[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg2[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_2[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_2[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_2[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg3[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_3[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg4[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_5[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_5[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%2]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_5[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg5[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_6[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg0[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg1[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_0[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg2[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_2[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_2[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_2[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg3[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_3[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg4[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_5[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_5[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%2]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_5[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg5[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_6[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before MergeSCFPass (merge-scf) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %dim = memref.dim %arg0, %c0 : memref<?x2xi32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg0[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_0 = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x4xf32>
  %0 = arith.muli %dim, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%0) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg1[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_0[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_1 = memref.dim %arg2, %c0 : memref<?x2xi32>
  %alloc_2 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg2[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_2[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_2[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_3 = memref.alloc(%dim_1) {alignment = 128 : i64} : memref<?x4xf32>
  %1 = arith.muli %dim_1, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%1) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%1]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_2[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg3[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_3[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  %dim_4 = memref.dim %arg4, %c0 : memref<?x2xi32>
  %alloc_5 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?xi32>
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%4] : memref<?xi32>
      scf.yield
    }
    scf.yield
  }
  scf.parallel (%arg6) = (%c0) to (%dim_4) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %5 = memref.load %arg4[%4, %arg8] : memref<?x2xi32>
        %6 = memref.load %alloc_5[%4] : memref<?xi32>
        %7 = arith.addi %6, %5 : i32
        memref.store %7, %alloc_5[%4] : memref<?xi32>
      }
      scf.yield
    }
    scf.yield
  }
  %alloc_6 = memref.alloc(%dim_4) {alignment = 128 : i64} : memref<?x4xf32>
  %2 = arith.muli %dim_4, %c4 : index
  scf.parallel (%arg6) = (%c0) to (%2) step (%c512) {
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%2]
    scf.parallel (%arg7) = (%c0) to (%3) step (%c1) {
      %4 = arith.addi %arg7, %arg6 : index
      %5 = arith.remsi %4, %c4 : index
      %6 = arith.divsi %4, %c4 : index
      %7 = memref.load %alloc_5[%6] : memref<?xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg5[%8, %5] : memref<?x4xf32>
      memref.store %9, %alloc_6[%6, %5] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After MergeSCFPass (merge-scf) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
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
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%17] : memref<?xi32>
      scf.yield
    }
    %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg0[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc[%17] : memref<?xi32>
      }
      scf.yield
    }
    %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg1[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_0[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%17] : memref<?xi32>
      scf.yield
    }
    %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg2[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc_2[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc_2[%17] : memref<?xi32>
      }
      scf.yield
    }
    %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%3]
    scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc_2[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg3[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_3[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    %14 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%14) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%17] : memref<?xi32>
      scf.yield
    }
    %15 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%15) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg4[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc_5[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc_5[%17] : memref<?xi32>
      }
      scf.yield
    }
    %16 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%6]
    scf.parallel (%arg7) = (%c0) to (%16) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc_5[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg5[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_6[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
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
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%17] : memref<?xi32>
      scf.yield
    }
    %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg0[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc[%17] : memref<?xi32>
      }
      scf.yield
    }
    %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg1[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_0[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%17] : memref<?xi32>
      scf.yield
    }
    %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg2[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc_2[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc_2[%17] : memref<?xi32>
      }
      scf.yield
    }
    %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%3]
    scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc_2[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg3[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_3[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    %14 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%14) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%17] : memref<?xi32>
      scf.yield
    }
    %15 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%15) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg4[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc_5[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc_5[%17] : memref<?xi32>
      }
      scf.yield
    }
    %16 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%6]
    scf.parallel (%arg7) = (%c0) to (%16) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc_5[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg5[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_6[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
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
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%17] : memref<?xi32>
      scf.yield
    }
    %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg0[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc[%17] : memref<?xi32>
      }
      scf.yield
    }
    %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg1[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_0[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%17] : memref<?xi32>
      scf.yield
    }
    %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg2[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc_2[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc_2[%17] : memref<?xi32>
      }
      scf.yield
    }
    %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%3]
    scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc_2[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg3[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_3[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    %14 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%14) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%17] : memref<?xi32>
      scf.yield
    }
    %15 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%15) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg4[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc_5[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc_5[%17] : memref<?xi32>
      }
      scf.yield
    }
    %16 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%6]
    scf.parallel (%arg7) = (%c0) to (%16) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc_5[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg5[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_6[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump Before CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
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
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc[%17] : memref<?xi32>
      scf.yield
    }
    %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim]
    scf.parallel (%arg7) = (%c0) to (%9) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg0[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc[%17] : memref<?xi32>
      }
      scf.yield
    }
    %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%0]
    scf.parallel (%arg7) = (%c0) to (%10) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg1[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_0[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%11) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_2[%17] : memref<?xi32>
      scf.yield
    }
    %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_1]
    scf.parallel (%arg7) = (%c0) to (%12) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg2[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc_2[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc_2[%17] : memref<?xi32>
      }
      scf.yield
    }
    %13 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%3]
    scf.parallel (%arg7) = (%c0) to (%13) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc_2[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg3[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_3[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    %14 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%14) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      memref.store %c0_i32, %alloc_5[%17] : memref<?xi32>
      scf.yield
    }
    %15 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%dim_4]
    scf.parallel (%arg7) = (%c0) to (%15) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      scf.for %arg8 = %c0 to %c2 step %c1 {
        %18 = memref.load %arg4[%17, %arg8] : memref<?x2xi32>
        %19 = memref.load %alloc_5[%17] : memref<?xi32>
        %20 = arith.addi %19, %18 : i32
        memref.store %20, %alloc_5[%17] : memref<?xi32>
      }
      scf.yield
    }
    %16 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg6)[%6]
    scf.parallel (%arg7) = (%c0) to (%16) step (%c1) {
      %17 = arith.addi %arg7, %arg6 : index
      %18 = arith.remsi %17, %c4 : index
      %19 = arith.divsi %17, %c4 : index
      %20 = memref.load %alloc_5[%19] : memref<?xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg5[%21, %18] : memref<?x4xf32>
      memref.store %22, %alloc_6[%19, %18] : memref<?x4xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
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

// -----// IR Dump Before FuseKernelLaunchPass (fuse-kernel-launch) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
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
      }
      %10 = affine.min #map(%arg6)[%dim_1]
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
      }
      %12 = affine.min #map(%arg6)[%dim_4]
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
      }
      scf.yield
    }
    return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump After FuseKernelLaunchPass (fuse-kernel-launch) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
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
      }
      %10 = affine.min #map(%arg6)[%dim_1]
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
      }
      %12 = affine.min #map(%arg6)[%dim_4]
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
      }
      scf.yield
    }
    return %alloc_0, %alloc_3, %alloc_6 : memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>
  }
}


// -----// IR Dump Before Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
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

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
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

// -----// IR Dump Before CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
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

// -----// IR Dump After CSE (cse) //----- //
func.func @cuda_kernel_0(%arg0: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "7:0", input.has_one_use = true}, %arg1: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "4:0", input.has_one_use = true}, %arg2: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "8:0", input.has_one_use = true}, %arg3: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "5:0", input.has_one_use = true}, %arg4: memref<?x2xi32> {input.fake_symbolic_shape = #tf_type.shape<137x2>, input.from = "9:0", input.has_one_use = true}, %arg5: memref<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<137x4>, input.from = "6:0", input.has_one_use = true}) -> (memref<?x4xf32>, memref<?x4xf32>, memref<?x4xf32>) attributes {BatchComputeFusion, _nano_compiler_group_idx = 0 : i64, _nano_compiler_kernel_size = 1 : i64, _nano_compiler_op_idx = "10", _nano_compiler_v2, batch_compute_pattern = "FuseGatherV2OpPattern", llvm.emit_c_interface, tf_entry} {
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

