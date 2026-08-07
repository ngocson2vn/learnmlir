# memref.subview
Given `%arg0: memref<?x576xf32>`,
```MLIR
%dim = memref.dim %arg0, %c0 : memref<?x576xf32>
%subview_11 = memref.subview %arg0[0, 256] [%dim, 6] [1, 1] : memref<?x576xf32> to memref<?x6xf32, strided<[576, 1], offset: 256>>
```

**This MLIR extracts a contiguous column-slice (a sub-matrix) from a dynamically-sized 2-D memref.**

### 1. Getting the dynamic dimension
```mlir
%dim = memref.dim %arg0, %c0 : memref<?x576xf32>
```
- `%arg0` has type `memref<?x576xf32>` → a 2-D buffer whose first dimension is dynamic and whose second dimension is fixed at 576 (`f32` elements).
- `memref.dim …, %c0` queries the size of dimension 0.
- Consequently `%dim` holds the runtime height of the matrix.

### 2. Creating the subview
```mlir
%subview_11 = memref.subview %arg0
    [0, 256]          // offsets
    [%dim, 6]         // sizes
    [1, 1]            // strides
  : memref<?x576xf32>
  to memref<?x6xf32, strided<[576, 1], offset: 256>>
```

`memref.subview` produces a *view* (no data is copied) that describes a rectangular region of the original buffer.

| Argument | Meaning |
|----------|---------|
| Offsets `[0, 256]` | Start at row 0, column 256 |
| Sizes `[%dim, 6]` | Take the whole height and exactly 6 consecutive columns |
| Strides `[1, 1]` | Step by 1 in both dimensions (i.e. the view itself is dense) |

### Resulting layout
The result type
```mlir
memref<?x6xf32, strided<[576, 1], offset: 256>>
```
encodes exactly how the view sits inside the original storage:

- Shape: `? × 6`
- Stride of the outer dimension = 576 → moving one row still advances by a full original row (the classic “row-major” stride).
- Stride of the inner dimension = 1 → the six columns are contiguous.
- Base offset = 256 → the first element of the view is located 256 elements into the underlying buffer (the start of column 256 of the first row).

### What the code is doing in plain English
It selects the slice

```
%arg0[0 : %dim, 256 : 262]
```

i.e. every row, but only the six columns that begin at index 256, and packages that region as a new memref of shape `?×6` that still points into the original storage.

### Elaborate more on strides `[1, 1]`
>Strides [1, 1] Step by 1 in both dimensions (i.e. the view itself is dense)
The strides argument in `memref.subview` is often the most confusing part. Let’s clarify it carefully.

### The three lists in `memref.subview`

```mlir
memref.subview %source
    [offsets...]     // where to start
    [sizes...]       // how many elements to take
    [strides...]     // how to step while taking them
```

The **strides** list tells the operation *how to walk* through the source memref when building the view.

- A stride of `1` in a dimension means “take consecutive elements” (no skipping).
- A stride of `2` would mean “take every other element”, etc.

In your example the strides are `[1, 1]`:

```mlir
memref.subview %arg0
    [0, 256]      // start at row 0, column 256
    [%dim, 6]     // take %dim rows and 6 columns
    [1, 1]        // step by 1 in both dimensions
```

This means:

- When moving to the next row of the view → move **1** row in the original buffer.
- When moving to the next column of the view → move **1** column in the original buffer.

So the view simply takes a contiguous rectangular block; it does **not** skip any rows or columns.

### Why the result type is still `strided<[576, 1]>`

Even though we asked for strides `[1, 1]`, the **layout** of the resulting memref is:

```mlir
memref<?x6xf32, strided<[576, 1], offset: 256>>
```

This is expected. Here’s why:

The original buffer has shape `? × 576`.  
When you take only 6 consecutive columns out of a row that is 576 elements wide, the elements of one row of the *view* are still contiguous, but the next row of the *view* starts 576 elements later in memory (the full original row width).

So:

| View dimension | Stride in the underlying buffer | Meaning |
|----------------|----------------------------------|---------|
| outer (rows)   | 576                              | jump a whole original row |
| inner (cols)   | 1                                | consecutive elements |

The `[1, 1]` you wrote is only the *relative* stepping with respect to the source dimensions. The compiler then computes the *absolute* strides that appear in the result type.

### Quick mental picture

```
Original row:  [ 0 1 2 ... 255 | 256 257 258 259 260 261 | 262 ... 575 ]
                                 ↑________________________↑
                                      taken by the view
```

Because the taken columns sit inside a wider row, the view is **not** a densely packed `? × 6` block in memory. It is a strided view with outer stride 576.

### Summary

- `strides = [1, 1]` → “don’t skip any elements while extracting the rectangle”.
- The result type `strided<[576, 1]>` → “in the underlying buffer this rectangle still has a large gap between rows”.

That is the distinction.
