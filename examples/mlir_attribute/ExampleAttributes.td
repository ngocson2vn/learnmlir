#ifndef EXAMPLE_ATTR_TD
#define EXAMPLE_ATTR_TD

include "mlir/IR/AttrTypeBase.td"
include "ExampleDialect.td"

// Base class for all Example attributes.
class Example_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<Example_Dialect, name, traits> {

  let description = [{
Base class for all Example attributes.
  }];

  let attrName = "example." # attrMnemonic;

  let mnemonic = attrMnemonic;

  code extraBaseClassDeclaration = [{
  }];

  let hasCustomAssemblyFormat = 1;
}

def BlockedEncoding : Example_Attr<"BlockedEncoding", "blocked"> {
  let description = [{
The blocked encoding for RankedTensorType
  }];

  let parameters = (ins
    "int":$rank
  );
}

#endif
