"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<i32>):
    %0 = "memref.alloc"() {operand_segment_sizes = array<i32: 0, 0>} : () -> memref<memref<i32>>
    "lmhlo.constant"(%0) {value = dense<21> : memref<i32>} : (memref<memref<i32>>) -> ()
    %1 = "memref.alloc"() {operand_segment_sizes = array<i32: 0, 0>} : () -> memref<memref<i32>>
    "lmhlo.add"(%0, %0, %1) : (memref<memref<i32>>, memref<memref<i32>>, memref<memref<i32>>) -> ()
    "func.return"(%1) : (memref<memref<i32>>) -> ()
  }) {function_type = (memref<i32>) -> (), sym_name = "main"} : () -> ()
}) : () -> ()
