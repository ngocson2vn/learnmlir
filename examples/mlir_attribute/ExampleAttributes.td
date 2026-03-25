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
  let genVerifyDecl = 1;
}

def BlockedEncoding : Example_Attr<"BlockedEncoding", "blocked"> {
  let description = [{
The blocked encoding for RankedTensorType. 
This encoding acts as a permanent label. It tells all downstream compiler passes: 
"Hey, whenever you allocate memory or generate loops for this tensor, 
make sure you process it in 4x4 blocks because the hardware demands it."
  }];

  let parameters = (ins
    "int":$numBlocks
  );
}

def DistributedSharding : Example_Attr<"DistributedSharding", "dist.sharding"> {
  let description = [{
If you are training a massive AI model, a single tensor<8192x8192xf32> might not fit on one GPU. 
Compilers like XLA or those built on MLIR will use the encoding field to represent how the tensor is chopped up across multiple chips.

// Pseudo-code for a tensor distributed across 4 GPUs
tensor<8192x8192xf32, #dist.sharding<{devices = [0, 1, 2, 3], split_axis = 0}>>

  }];

  let parameters = (ins
    // A custom `template <typename T> llvm::hash_code hash_value(const std::vector<T>& vec)`
    // must be implemented in `std` namespace to make ADL work
    "std::vector<int>":$devices,
    "int":$splitAxis
  );
}

#endif
