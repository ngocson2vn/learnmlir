# useDefaultAttributePrinterParser
ExampleDialect.td
```MLIR
  // CRUCIAL: Tells the dialect to use the auto-generated dispatch hooks
  // for all attributes that define a `mnemonic` and (`assemblyFormat` or `hasCustomAssemblyFormat`).
  let useDefaultAttributePrinterParser = 1;
```

# `assemblyFormat` vs `hasCustomAssemblyFormat`
Read section [`assemblyFormat` vs `hasCustomAssemblyFormat`](../../docs/mlir_tablegen.md#assemblyformat-vs-hascustomassemblyformat)

# Encoding
See [RankedTensorType encoding](../../docs/mlir_tensor_encoding.md)
