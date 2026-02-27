module {
  func.func @main(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>) -> tensor<?xf64> {
    %0 = "toy.add"(%arg0, %arg1) : (tensor<?xf64>, tensor<?xf64>) -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }
}
