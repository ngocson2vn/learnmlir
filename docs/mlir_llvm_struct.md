# MLIR LLVM Struct
```MLIR
%28 = llvm.mlir.undef : !llvm.struct<(ptr<3>, i32, i32)>
```
In the context of MLIR’s LLVM Dialect, the syntax `!llvm.struct<(ptr<3>, i32, i32)>` describes a structure containing a pointer and two integers. The specific part you're asking about, `ptr<3>`, refers to a **pointer with a specific Address Space.**

Here is the breakdown of what that actually means:

## 1. The Pointer Type (`ptr`)
In modern MLIR (and LLVM), pointers are **opaque**. This means they no longer carry the type of the data they point to (like `ptr<i32>`). Instead, they just represent a memory address. 

## 2. The Address Space (`<3>`)
The number inside the angle brackets represents the **Address Space Identifier**. 

Most standard CPU programs use Address Space `0` (the default), which represents generic global memory. However, in specialized hardware—like GPUs or DSPs—memory is partitioned into different physical or logical regions. The integer `3` is a target-specific index that tells the compiler which "pool" of memory this pointer inhabits.

### Common Mapping Examples
While the meaning of `3` depends entirely on your compilation target (the chip you are building for), here are the most common conventions:

| Target | Address Space `3` Usually Represents |
| :--- | :--- |
| **NVIDIA GPU (NVPTX)** | **Shared Memory** (Fast, on-chip memory shared by a thread block) |
| **AMD GPU (AMDGPU)** | **Local Data Store (LDS)** |
| **Standard CPU** | Often unused, or reserved for specific security/instrumentation tiers |

## 3. Why is this important?
The compiler needs to know the address space for two main reasons:

* **Instruction Selection:** Loading from Address Space `3` (Shared Memory) might require a different assembly instruction than loading from Address Space `0` (Global Memory).
* **Optimizations:** The compiler can assume that a pointer in `ptr<3>` cannot alias (overlap) with a pointer in `ptr<0>`, allowing for more aggressive performance optimizations.

---

### Summary
In your struct `!llvm.struct<(ptr<3>, i32, i32)>`, the first element is a pointer pointing to a memory region designated as **Address Space 3**. If you are working on GPU kernels, this almost certainly refers to **Local/Shared memory**.
