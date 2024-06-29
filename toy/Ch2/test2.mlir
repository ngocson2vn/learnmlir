module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64> loc("./test/test2.toy":1:1)) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64> loc("./test/test2.toy":2:20)
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64> loc("./test/test2.toy":2:10)
    toy.return %1 : tensor<*xf64> loc("./test/test2.toy":2:3)
  } loc("./test/test2.toy":1:1)
} loc(unknown)
