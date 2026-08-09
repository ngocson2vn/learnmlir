// -----// IR Dump After ComputeOpAndFuncBufferizePass (computeop-and-func-bufferize) ('builtin.module' operation) //----- //
#map = affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>
module {
  func.func @main(%arg0: memref<?x576xf32>, %arg1: memref<?x30xi64>) -> memref<?x30xf32> attributes {llvm.emit_c_interface, tf_entry} {
    %0 = bufferization.to_tensor %arg1 : memref<?x30xi64> to tensor<?x30xi64>
    %1 = bufferization.to_tensor %arg0 : memref<?x576xf32> to tensor<?x576xf32>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %1, %c0 : tensor<?x576xf32>
    %2 = shape.shape_of %0 : tensor<?x30xi64> -> tensor<2xindex>
    %3 = bufferization.to_buffer %2 : tensor<2xindex> to memref<2xindex>
    %4 = bufferization.to_buffer %2 : tensor<2xindex> to memref<2xindex>
    %extracted_slice = tensor.extract_slice %1[0, 256] [%dim, 6] [1, 1] : tensor<?x576xf32> to tensor<?x6xf32>
    %extracted_slice_0 = tensor.extract_slice %extracted_slice[0, 0] [%dim, 1] [1, 1] : tensor<?x6xf32> to tensor<?x1xf32>
    %5 = bufferization.to_buffer %extracted_slice_0 : tensor<?x1xf32> to memref<?x1xf32>
    %c0_1 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    %6 = arith.muli %c1, %c1_2 : index
    %c0_3 = arith.constant 0 : index
    %dim_4 = memref.dim %5, %c0_3 : memref<?x1xf32>
    %c0_5 = arith.constant 0 : index
    %7 = memref.load %4[%c0_5] : memref<2xindex>
    %8 = arith.cmpi slt, %dim_4, %7 : index
    %9 = arith.select %8, %c0_1, %6 : index
    %c1_6 = arith.constant 1 : index
    %10 = memref.load %3[%c1_6] : memref<2xindex>
    %11 = arith.cmpi slt, %c1_2, %10 : index
    %12 = arith.select %11, %c0_1, %c1 : index
    %reinterpret_cast = memref.reinterpret_cast %5 to offset: [0], sizes: [%7, 30], strides: [%9, %12] : memref<?x1xf32> to memref<?x30xf32, #map>
    %13 = bufferization.to_tensor %reinterpret_cast : memref<?x30xf32, #map> to tensor<?x30xf32>
    %14 = bufferization.to_buffer %13 : tensor<?x30xf32> to memref<?x30xf32>
    return %14 : memref<?x30xf32>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module {
  func.func @main(%arg0: memref<?x576xf32>, %arg1: memref<?x30xi64>) -> memref<?x30xf32> attributes {llvm.emit_c_interface, tf_entry} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x576xf32> to tensor<?x576xf32>
    %dim = memref.dim %arg0, %c0 : memref<?x576xf32>
    %extracted_slice = tensor.extract_slice %0[0, 256] [%dim, 6] [1, 1] : tensor<?x576xf32> to tensor<?x6xf32>
    %extracted_slice_0 = tensor.extract_slice %extracted_slice[0, 0] [%dim, 1] [1, 1] : tensor<?x6xf32> to tensor<?x1xf32>
    %1 = bufferization.to_buffer %extracted_slice_0 : tensor<?x1xf32> to memref<?x1xf32>
    %dim_1 = memref.dim %arg1, %c0 : memref<?x30xi64>
    %2 = arith.cmpi slt, %dim, %dim_1 : index
    %3 = arith.select %2, %c0, %c1 : index
    %reinterpret_cast = memref.reinterpret_cast %1 to offset: [0], sizes: [%dim_1, 30], strides: [%3, 0] : memref<?x1xf32> to memref<?x30xf32, strided<[?, 0]>>
    %dim_2 = memref.dim %reinterpret_cast, %c0 : memref<?x30xf32, strided<[?, 0]>>
    %alloc = memref.alloc(%dim_2) : memref<?x30xf32>
    memref.copy %reinterpret_cast, %alloc : memref<?x30xf32, strided<[?, 0]>> to memref<?x30xf32>
    return %alloc : memref<?x30xf32>
  }
}


// -----// IR Dump After FinalBufferizePass (final-bufferize) ('builtin.module' operation) //----- //
module {
  func.func @main(%arg0: memref<?x576xf32>, %arg1: memref<?x30xi64>) -> memref<?x30xf32> attributes {llvm.emit_c_interface, tf_entry} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x576xf32>
    %subview = memref.subview %arg0[0, 256] [%dim, 6] [1, 1] : memref<?x576xf32> to memref<?x6xf32, strided<[576, 1], offset: 256>>
    %subview_0 = memref.subview %subview[0, 0] [%dim, 1] [1, 1] : memref<?x6xf32, strided<[576, 1], offset: 256>> to memref<?x1xf32, strided<[576, 1], offset: 256>>
    %c0_1 = arith.constant 0 : index
    %dim_2 = memref.dim %subview_0, %c0_1 : memref<?x1xf32, strided<[576, 1], offset: 256>>
    %alloc = memref.alloc(%dim_2) {alignment = 64 : i64} : memref<?x1xf32>
    memref.copy %subview_0, %alloc : memref<?x1xf32, strided<[576, 1], offset: 256>> to memref<?x1xf32>
    %dim_3 = memref.dim %arg1, %c0 : memref<?x30xi64>
    %0 = arith.cmpi slt, %dim, %dim_3 : index
    %1 = arith.select %0, %c0, %c1 : index
    %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [%dim_3, 30], strides: [%1, 0] : memref<?x1xf32> to memref<?x30xf32, strided<[?, 0]>>
    %dim_4 = memref.dim %reinterpret_cast, %c0 : memref<?x30xf32, strided<[?, 0]>>
    %alloc_5 = memref.alloc(%dim_4) : memref<?x30xf32>
    memref.copy %reinterpret_cast, %alloc_5 : memref<?x30xf32, strided<[?, 0]>> to memref<?x30xf32>
    return %alloc_5 : memref<?x30xf32>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
module {
  func.func @main(%arg0: memref<?x576xf32>, %arg1: memref<?x30xi64>) -> memref<?x30xf32> attributes {llvm.emit_c_interface, tf_entry} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x576xf32>
    %subview = memref.subview %arg0[0, 256] [%dim, 6] [1, 1] : memref<?x576xf32> to memref<?x6xf32, strided<[576, 1], offset: 256>>
    %subview_0 = memref.subview %subview[0, 0] [%dim, 1] [1, 1] : memref<?x6xf32, strided<[576, 1], offset: 256>> to memref<?x1xf32, strided<[576, 1], offset: 256>>
    %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<?x1xf32>
    memref.copy %subview_0, %alloc : memref<?x1xf32, strided<[576, 1], offset: 256>> to memref<?x1xf32>
    %dim_1 = memref.dim %arg1, %c0 : memref<?x30xi64>
    %0 = arith.cmpi slt, %dim, %dim_1 : index
    %1 = arith.select %0, %c0, %c1 : index
    %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [%dim_1, 30], strides: [%1, 0] : memref<?x1xf32> to memref<?x30xf32, strided<[?, 0]>>
    %dim_2 = memref.dim %reinterpret_cast, %c0 : memref<?x30xf32, strided<[?, 0]>>
    %alloc_3 = memref.alloc(%dim_2) : memref<?x30xf32>
    memref.copy %reinterpret_cast, %alloc_3 : memref<?x30xf32, strided<[?, 0]>> to memref<?x30xf32>
    return %alloc_3 : memref<?x30xf32>
  }
}


