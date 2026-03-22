# MLIR Attributes and Types
https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/#attributes-and-types

The C++ Attribute and Type classes in MLIR (like Ops, and many other things) are value-typed. This means that instances of Attribute or Type are passed around by-value, as opposed to by-pointer or by-reference. The Attribute and Type classes act as wrappers around internal storage objects that are uniqued within an instance of an MLIRContext.