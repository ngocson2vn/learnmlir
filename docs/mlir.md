# MLIR Language Reference
https://mlir.llvm.org/docs/LangRef/

MLIR is designed to be used in three different forms: a human-readable textual form suitable for debugging, an in-memory form suitable for programmatic transformations and analysis, and a compact serialized form suitable for storage and transport. The different forms all describe the same semantic content. This document describes the human-readable textual form.

## High-Level Structure
nodes == Operations
edges == Values

## Notation
This document describes the grammar using Extended Backus-Naur Form (EBNF).

# Notes
Any Language -> Dialect + MLIR -> MLIR IR -> Analyze -> Transform/Optimize -> MLIR IR
