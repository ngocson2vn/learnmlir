module {
  func.func @main() {
    %1 = "arith.constant"() {value = dense<[1.0, 2.0]> : tensor<2xf64>} : () -> tensor<2xf64>
    %2 = "arith.constant"() {value = dense<[3.0, 4.0]> : tensor<2xf64>} : () -> tensor<2xf64>
    %3 = "toy.add"(%1, %2) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return
  }
}
