# Chapter 3

https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/

# MLIR Notes
- We divide compiler transformations into two categories: local and global.

- We also need to update our main file, toyc.cpp, to add an **optimization pipeline**. In MLIR, the optimizations are run through a `PassManager` in a similar way to LLVM: