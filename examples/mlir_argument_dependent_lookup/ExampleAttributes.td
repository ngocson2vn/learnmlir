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

def DistributedSharding : Example_Attr<"DistributedSharding", "dist.sharding"> {
  let description = [{
If you are training a massive AI model, a single tensor<8192x8192xf32> might not fit on one GPU. 
Compilers like XLA or those built on MLIR will use the encoding field to represent how the tensor is chopped up across multiple chips.

// Pseudo-code for a tensor distributed across 4 GPUs
tensor<8192x8192xf32, #dist.sharding<{devices = [0, 1, 2, 3], split_axis = 0}>>

  }];

  let parameters = (ins
    "std::vector<int>":$devices,
    "int":$splitAxis
  );
}

#endif
