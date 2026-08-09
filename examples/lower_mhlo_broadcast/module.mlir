module {
  func.func @main(%arg0: tensor<?x576xf32>, %arg1: tensor<?x30xi64>) -> (tensor<?x30xf32>) attributes {llvm.emit_c_interface, tf_entry} {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x576xf32>
    %0 = shape.shape_of %arg1 : tensor<?x30xi64> -> tensor<2xindex>
    %extracted_slice = tensor.extract_slice %arg0[0, 256] [%dim, 6] [1, 1] : tensor<?x576xf32> to tensor<?x6xf32>
    %extracted_slice_0 = tensor.extract_slice %extracted_slice[0, 0] [%dim, 1] [1, 1] : tensor<?x6xf32> to tensor<?x1xf32>
    %1 = "mhlo.dynamic_broadcast_in_dim"(%extracted_slice_0, %0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x30xf32>

    return %1 : tensor<?x30xf32>
  }
}
