// ExampleDialect.td
include "mlir/IR/OpBase.td"
include "ExampleTraits.td"

// 1. Define the Dialect
def Example_Dialect : Dialect {
  let name = "example";
  let cppNamespace = "::mlir::example";
  let summary = "An example dialect for demonstrating interfaces.";
}

// Base class for operations in this dialect
class Example_Op<string mnemonic, list<Trait> traits = []> :
    Op<Example_Dialect, mnemonic, traits>;

// 2. Define an Operation that uses the DEFAULT interface method
// We simply attach the interface without declaring the methods.
def DefaultOp : Example_Op<"default", [VerifyMagicNumberTrait]> {
  let summary = "An operation that uses the default interface method.";
}

// 3. Define an Operation that OVERRIDES the interface method
// Using `DeclareOpInterfaceMethods` automatically declares the C++ method 
// `int getMagicNumber();` in the generated Op class.
def CustomOp : Example_Op<"custom", [VerifyMagicNumberTrait]> {
  let summary = "A custom operation that overrides the interface method.";
}
