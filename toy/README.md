# Toy Tutorial

This contains sample code to support the tutorial on using MLIR for
building a compiler for a simple Toy language.

See https://mlir.llvm.org/docs/Tutorials/Toy/

## Chapter 2
### How to build
```Bash
./build.sh Ch2
```

### How to run
```Bash
./run2.sh
```

### Flow Diagram
<img src="./Ch2/images/toy.png" width="70%" alt="Toy MLIR Flow Diagram" />

- `ToyDialect` is built on top of MLIR
- Toy Compiler loads `ToyDialect` into MLIR
- Toy Compiler calls `Lexer` and `Parser` to produce Toy AST
- Toy Compiler calls `MLIRGenImpl` methods to generate Toy IR in MLIR format

### How does MLIR load `ToyDialect`?
