module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("./test/sample.toy":4:11)
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<*xf64> loc("./test/sample.toy":5:11)
    toy.print %1 : tensor<*xf64> loc("./test/sample.toy":6:3)
    toy.return loc("./test/sample.toy":1:1)
  } loc("./test/sample.toy":1:1)
} loc(unknown)
