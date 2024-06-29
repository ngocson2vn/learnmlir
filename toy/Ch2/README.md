# Chapter 2
https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/

## How to build
```Bash
./build.sh
```

## How to run
```Bash
./run.sh
```

## Flow Diagram
<img src="./images/toy.png" width="70%" alt="Toy MLIR Flow Diagram" />

- `ToyDialect` is built on top of MLIR
- Toy Compiler loads `ToyDialect` into MLIR
- Toy Compiler calls `Lexer` and `Parser` to produce Toy AST
- Toy Compiler calls `MLIRGenImpl` methods to generate Toy IR in MLIR format

## How does MLIR load `ToyDialect`?

# MLIR Notes
Understand MLIR language --> How to analyze MLIR --> How to transform MLIR
## Operations
https://mlir.llvm.org/docs/LangRef/#operations

In MLIR, **Operations** are the core unit of abstraction and computation, similar in many ways to LLVM instructions. Operations can have application-specific semantics and can be used to represent all of the core IR structures in LLVM: instructions, globals (like functions), modules, etc.

@FIXME: what are SSA values?  
**Static single-assignment form**  
In compiler design, static single assignment form (often abbreviated as SSA form or simply SSA) is a type of intermediate representation (IR) where **each variable is assigned exactly once**.

**Operations** are modeled using a small set of concepts, enabling operations to be reasoned about and manipulated generically. These concepts are:
- A name for the operation.
- A list of SSA operand values.
- A list of attributes.
- A list of types for result values.
- A source location for debugging purposes.
- A list of successors blocks (for branches, mostly).
- A list of regions (for structural operations like functions).