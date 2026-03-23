// ExampleInterface.td
include "mlir/IR/OpBase.td"

def ExampleOpInterface : OpInterface<"ExampleOpInterface"> {
  let description = [{
    An example interface for operations to retrieve a magic number.
  }];
  let cppNamespace = "::mlir::example";

  let methods = [
    InterfaceMethod<
      /*description=*/"Returns the magic number of the operation.",
      /*retTy=*/"int",
      /*methodName=*/"getMagicNumber",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        // Default implementation if an operation doesn't override it.
        return 42; 
      }]
    >
  ];
}
