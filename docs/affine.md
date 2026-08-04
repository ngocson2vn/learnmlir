# linalg.generic
## 1. Iteration space
**In a `linalg.generic`, the iteration space is not declared explicitly.** It is **derived** from the shapes of the operands together with the `indexing_maps`.

### Official rule (Property 1 of Linalg)

> **Input and Output Operands Define The Iteration Space**  
> A `linalg.generic` op fully *derives* the specification of its iteration space from its operands.

This is a core design principle of the Linalg dialect.

### How the derivation works (applied to the example)

```mlir
#identity = affine_map<(d0, d1) -> (d0, d1)>

linalg.generic {
  indexing_maps = [#identity, #identity, #identity],
  iterator_types = ["parallel", "parallel"]
} ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
  outs(%C : tensor<?x?xf32>) { ... }
```

1. **Rank of the iteration space**  
   Taken from the domain of the indexing maps (here `(d0, d1)`) and from the length of `iterator_types`.  
   → 2-dimensional iteration space.

2. **Bounds / sizes of each dimension**  
   Inferred by looking at which tensor dimensions are accessed by each iteration dimension, using the indexing maps.

   Because every map is the **identity**:
   - Iteration dimension `d0` maps to dimension 0 of `%A`, `%B` and `%C`
   - Iteration dimension `d1` maps to dimension 1 of `%A`, `%B` and `%C`

   Therefore the size of the iteration space is exactly the shape of the tensors:
   ```text
   iteration space = [0 .. dim(%A, 0)) × [0 .. dim(%A, 1))
   ```
   (The shapes of `%A`, `%B` and `%C` are required to be consistent; otherwise the IR is invalid.)

3. **Result of the inference**  
   The op behaves as if it had been written with two nested parallel loops whose bounds come from the tensor dimensions:

   ```mlir
   // conceptual lowering
   for %d0 = 0 to tensor.dim %A, 0 {
     for %d1 = 0 to tensor.dim %A, 1 {
       // body of the generic
     }
   }
   ```

### General case (when maps are not identity)

The same principle applies, but the correspondence is obtained by “inverting” the indexing maps:

- Each iteration dimension is associated with the tensor dimension(s) it indexes.
- The size of that iteration dimension is taken from the corresponding operand dimension.
- All operands that contribute a size for the same iteration dimension must agree.

This is why Linalg can guarantee (by construction) that there are no out-of-bounds accesses, provided the dynamic dimensions of the operands are consistent.

### Summary for the given example

| Component              | Role in determining the iteration space                  |
|------------------------|----------------------------------------------------------|
| `iterator_types`       | Declares the **rank** and the nature of each dimension   |
| `indexing_maps`        | Defines the **mapping** from iteration dims → tensor dims|
| Shapes of `%A/%B/%C`   | Supply the actual **bounds** of each iteration dimension |

The iteration space is therefore completely determined by the combination of the operand shapes and the indexing maps; there is no separate loop-bound attribute on the `linalg.generic` itself.

## 2. affine_map
So the main purpose of affine_map<(d0, d1) -> (d0, d1)> is to compute indices of input tensors?

**Not only for input tensors — it defines the indexing for *every* operand (both `ins` and `outs`).**

### Precise role of the affine map

In a `linalg.generic`, each entry in `indexing_maps` answers this question:

> Given a point `(d0, d1, …)` in the **iteration space**, which element of *this particular operand* should be accessed?

So the map `affine_map<(d0, d1) -> (d0, d1)>` means:

```text
iteration point (d0, d1)  →  access operand[d0, d1]
```

It is used for **both**:

- **Inputs** (`ins`) → to decide which value to *read*
- **Outputs** (`outs`) → to decide which location to *write* (or read-modify-write in reductions)

### In the concrete example

```mlir
indexing_maps = [
  #identity,   // for %A  (input)
  #identity,   // for %B  (input)
  #identity    // for %C  (output)
]
```

All three maps being the identity simply says:

- Read `A[d0, d1]`
- Read `B[d0, d1]`
- Write the result into `C[d0, d1]`

### Broader purpose

The affine maps are the **bridge** between:

1. The abstract iteration space (defined by the shapes of the operands), and  
2. The concrete memory/tensor accesses performed by the computation.

They are what make Linalg “structured”: the compiler can reason about data movement, tiling, fusion, vectorization, etc., without having to analyze arbitrary index arithmetic inside a loop nest.

### Short correction of your statement

| Your phrasing                                      | More accurate view                                      |
|----------------------------------------------------|---------------------------------------------------------|
| “compute indices of input tensors”                 | compute indices of **all operands** (inputs *and* outputs) |
| main purpose                                       | define the **access pattern** of each operand relative to the iteration space |

So yes — computing indices is the immediate effect — but the deeper purpose is to *declare the relationship* between iteration space and data layout for every operand.

# Explain this MLIR IR:
```mlir
%15 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg7)[%c1, %c0]
```

Let’s break down this MLIR (Multi-Level Intermediate Representation) operation step by step to understand what it does.
This is an affine.apply operation in MLIR, which applies an affine transformation (a linear function) to its input operands based on an affine map. The result is stored in %15.


Components of the Operation <br/>
## Affine Map:
```mlir
affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
```
An affine map defines a mathematical transformation. It takes inputs and produces outputs based on a formula.<br/>
(d0) represents the input dimension (a single input value in this case).<br/>
[s0, s1] are symbolic parameters (constants or variables provided at runtime).<br/>
-> (d0 * s0 + s1) is the output expression, meaning the result is computed as d0 * s0 + s1.<br/>
In plain terms, this map multiplies the input d0 by s0 and then adds s1.<br/>
Operands:<br/>
(%arg7): This is the input value, corresponding to d0 in the affine map. It’s likely a variable or value defined elsewhere in the IR (e.g., an argument to a function).<br/>
[%c1, %c0]: These are the symbolic arguments corresponding to [s0, s1] in the affine map.<br/>
%c1: A constant or value representing s0.<br/>
%c0: A constant or value representing s1.<br/>
In MLIR, %cN typically denotes a constant (e.g., %c1 might be the integer 1, and %c0 might be 0), but their exact values depend on the surrounding context or definitions.<br/>
Result:<br/>
%15: This is the output of the operation, the result of applying the affine map to the input %arg7 with the symbolic values %c1 and %c0.<br/>

## What It Does
Substituting the operands into the affine map:<br/>
d0 is %arg7.<br/>
s0 is %c1.<br/>
s1 is %c0.<br/>
The affine map (d0 * s0 + s1) becomes:<br/>
```mlir
%15 = %arg7 * %c1 + %c0
```
This is a simple linear transformation of %arg7. For example:<br/>
If %c1 is 1 and %c0 is 0, then %15 = %arg7 * 1 + 0 = %arg7 (identity transformation).<br/>
If %c1 is 2 and %c0 is 3, then %15 = %arg7 * 2 + 3 (doubling %arg7 and adding 3).<br/>

## Intuitive Explanation
Think of affine.apply as a way to perform index computations or transformations commonly used in compilers (e.g., for loop bounds or memory addressing). Here, it’s taking an input value (%arg7), scaling it by some factor (%c1), and offsetting it by another value (%c0). The exact meaning depends on what %arg7, %c1, and %c0 represent in the broader program—likely something like a loop index or array offset.<br/><br/>

Context in MLIR<br/>
In MLIR’s affine dialect, such operations are often used to represent structured computations (e.g., in loops or tensor operations). Without more of the IR, we can’t say exactly what %arg7, %c1, or %c0 are, but this operation is a building block for defining how data is accessed or transformed.
Does that clarify it? If you have more of the IR or specific values for %c1 and %c0, I can refine the explanation further!

## Example
```MLIR
%20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3 + 3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x1x1x51xf16>) outs(%19 : tensor<?x4x50x50xf16>) { ^bb0(%in: f16, %out: f16): linalg.yield %in : f16 } -> tensor<?x4x50x50xf16>
```

**affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3 + 1)> is for calculating indices of the input tensor**

In a `linalg.generic` operation, the `indexing_maps` array pairs its maps to the input and output tensors strictly based on **order**.

Because `affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, d3 + 1)>` is the **first** map in the list, it serves as the coordinate calculator for the **first** input tensor (`%arg0`).

Think of it as a function that translates loop coordinates into memory coordinates. If the underlying 4D loop is currently at iteration `(d0=2, d1=1, d2=3, d3=5)`, the operation feeds those numbers into your input map:

* **Input Map:** `(2, 1, 3, 5) -> (2, 0, 0, 5 + 1)`
* **Calculated Input Index:** `[2, 0, 0, 6]`

The operation now knows it must go to the input tensor `%arg0` and read the exact value stored at `[2, 0, 0, 6]`. It will do this calculation for every single step of the loop.

Let's look at each dimension:

* **`d0 -> d0`:** The first dimension is mapped 1-to-1. If the outermost loop is at index `5`, it reads from index `5` of the input tensor's first dimension (typically the batch dimension).
* **`d1 -> 0` and `d2 -> 0` (Broadcasting):** This is where **broadcasting** happens. Even as the `d1` and `d2` loops iterate forward (e.g., `d1` goes `0, 1, 2, 3`), the input tensor *always* reads from index `0` for its second and third dimensions. Because the loops advance but the read coordinate stays frozen at `0`, the exact same input data is reused (or "broadcast") across the `d1` and `d2` dimensions of the output.
* **`d3 -> d3 + 1` (Slicing/Shifting):** This is an offset. When the innermost loop `d3` is at index `0`, it reads from index `1` in the input tensor. When `d3` is `1`, it reads from `2`, and so on. This effectively skips the very first element (index `0`) of the input's last dimension, creating a shifted "slice" of the data.

