# Lower mhlo.dynamic_broadcast_in_dim
This project is for reproducing a correctness bug when lowering `mhlo.dynamic_broadcast_in_dim` to `memref` dialect.

Input: [module.mlir](./module.mlir)

## Build
```bash
make
```

## Run
```bash
make run
```
Output: lowering.mlir
