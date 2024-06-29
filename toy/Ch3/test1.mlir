module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64> loc("./test/test1.toy":1:1)) -> tensor<*xf64> {
    toy.return %arg0 : tensor<*xf64> loc("./test/test1.toy":2:3)
  } loc("./test/test1.toy":1:1)
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc(fused["./test/test1.toy":6:3, "./test/test1.toy":6:17])
    %1 = toy.generic_call @transpose_transpose(%0) : (tensor<2x3xf64>) -> tensor<*xf64> loc("./test/test1.toy":7:12)
    toy.print %1 : tensor<*xf64> loc("./test/test1.toy":8:3)
    toy.return loc("./test/test1.toy":5:1)
  } loc("./test/test1.toy":5:1)
} loc(unknown)
