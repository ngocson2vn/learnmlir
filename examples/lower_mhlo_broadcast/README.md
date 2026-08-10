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

## Correctness
After ComputeOpAndFuncBufferizePass:
```mlir
module {
  func.func @main(%arg0: memref<?x576xf32>, %arg1: memref<?x30xi64>) -> memref<?x30xf32> attributes {llvm.emit_c_interface, tf_entry} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x576xf32> to tensor<?x576xf32>
    %dim = memref.dim %arg0, %c0 : memref<?x576xf32>
    %extracted_slice = tensor.extract_slice %0[0, 256] [%dim, 6] [1, 1] : tensor<?x576xf32> to tensor<?x6xf32>
    %extracted_slice_0 = tensor.extract_slice %extracted_slice[0, 0] [%dim, 1] [1, 1] : tensor<?x6xf32> to tensor<?x1xf32>
    %1 = bufferization.to_buffer %extracted_slice_0 : tensor<?x1xf32> to memref<?x1xf32>
    %dim_1 = memref.dim %arg1, %c0 : memref<?x30xi64>
    %2 = arith.cmpi slt, %dim, %dim_1 : index
    %3 = arith.select %2, %c0, %c1 : index
    %reinterpret_cast = memref.reinterpret_cast %1 to offset: [0], sizes: [%dim_1, 30], strides: [%3, 0] : memref<?x1xf32> to memref<?x30xf32, strided<[?, 0]>>
    %dim_2 = memref.dim %reinterpret_cast, %c0 : memref<?x30xf32, strided<[?, 0]>>
    %alloc = memref.alloc(%dim_2) : memref<?x30xf32>
    memref.copy %reinterpret_cast, %alloc : memref<?x30xf32, strided<[?, 0]>> to memref<?x30xf32>
    return %alloc : memref<?x30xf32>
  }
}
```
`memref.reinterpret_cast` causes 2 correctness bugs. Details: [../../docs/memref_reinterpret_cast.md](../../docs/memref_reinterpret_cast.md)


After FinalBufferizePass:
```mlir
module {
  func.func @main(%arg0: memref<?x576xf32>, %arg1: memref<?x30xi64>) -> memref<?x30xf32> attributes {llvm.emit_c_interface, tf_entry} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x576xf32>
    %subview = memref.subview %arg0[0, 256] [%dim, 6] [1, 1] : memref<?x576xf32> to memref<?x6xf32, strided<[576, 1], offset: 256>>
    %subview_0 = memref.subview %subview[0, 0] [%dim, 1] [1, 1] : memref<?x6xf32, strided<[576, 1], offset: 256>> to memref<?x1xf32, strided<[576, 1], offset: 256>>
    %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<?x1xf32>
    memref.copy %subview_0, %alloc : memref<?x1xf32, strided<[576, 1], offset: 256>> to memref<?x1xf32>
    %dim_1 = memref.dim %arg1, %c0 : memref<?x30xi64>
    %0 = arith.cmpi slt, %dim, %dim_1 : index
    %1 = arith.select %0, %c0, %c1 : index
    %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [%dim_1, 30], strides: [%1, 0] : memref<?x1xf32> to memref<?x30xf32, strided<[?, 0]>>
    %dim_2 = memref.dim %reinterpret_cast, %c0 : memref<?x30xf32, strided<[?, 0]>>
    %alloc_3 = memref.alloc(%dim_2) : memref<?x30xf32>
    memref.copy %reinterpret_cast, %alloc_3 : memref<?x30xf32, strided<[?, 0]>> to memref<?x30xf32>
    return %alloc_3 : memref<?x30xf32>
  }
}
```
The 2 correctness bugs are fixed by the explicit materialization of the extracted column.

**"explicit materialization of the extracted column"** means:

Turning the *logical view* of the selected column into a real, owned, contiguous buffer that actually contains those values.

### Before materialization
The two `memref.subview` operations produce only a **view**:

```mlir
memref<?x1xf32, strided<[576, 1], offset: 256>>
```

This does **not** own any new memory. It just describes how to reach the desired elements inside the original `%arg0` buffer (every 576th float, starting at offset 256). The data still lives in the original allocation and is non-contiguous.

### Materialization step
```mlir
%alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<?x1xf32>
memref.copy %subview_0, %alloc
```

This does two things:

1. Allocates a fresh contiguous buffer of the right size.
2. Copies the (scattered) values from the strided view into that new buffer.

After the copy, `%alloc` is a dense `memref<?x1xf32>` that *owns* the extracted column data. The values are now packed one after another in memory.

That dense buffer is what the later `reinterpret_cast` + broadcast logic operates on. The standard MHLO broadcast lowering assumes its input is contiguous; the explicit `alloc` + `copy` satisfies that assumption.