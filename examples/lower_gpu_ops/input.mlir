"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: index):
    %0 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %1 = "memref.alloca"() {operand_segment_sizes = array<i32: 0, 0>} : () -> memref<f32>
    %2 = "gpu.memset"(%0, %1, %0) : (f32, memref<f32>, f32) -> !gpu.async.token
    %3:2 = "gpu.alloc"(%2, %arg0) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!gpu.async.token, index) -> (memref<?xf32, 3 : i32>, !gpu.async.token)
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "kernel"} : () -> ()
}) : () -> ()
