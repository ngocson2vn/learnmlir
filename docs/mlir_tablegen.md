# Parameters
```MLIR
def TT_PtrType : TritonTypeDef<"Pointer", "ptr"> {
  let parameters = (ins "Type":$pointeeType, "int":$addressSpace);
}
```

TableGen will generate an internal storage class for this type, namely, `PointerTypeStorage` which has two data members:
```C++
struct PointerTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, int>;
  PointerTypeStorage(Type pointeeType, int addressSpace) : pointeeType(std::move(pointeeType)), addressSpace(std::move(addressSpace)) {}

  KeyTy getAsKey() const {
    return KeyTy(pointeeType, addressSpace);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (pointeeType == std::get<0>(tblgenKey)) && (addressSpace == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static PointerTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto pointeeType = std::move(std::get<0>(tblgenKey));
    auto addressSpace = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<PointerTypeStorage>()) PointerTypeStorage(std::move(pointeeType), std::move(addressSpace));
  }

  Type pointeeType;
  int addressSpace;
};
```

TableGen will also generate a class for this type which as a `get(Type pointeeType, int addressSpace)` method:
```C++
class PointerType : public ::mlir::Type::TypeBase<PointerType, ::mlir::Type, detail::PointerTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tt.ptr";
  static constexpr ::llvm::StringLiteral dialectName = "tt";
  static PointerType get(Type pointeeType, int addressSpace);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ptr"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getPointeeType() const;
  int getAddressSpace() const;
};
```

# Builders
```MLIR
def TT_PtrType : TritonTypeDef<"Pointer", "ptr"> {
    let builders = [
        TypeBuilderWithInferredContext<(ins
            "Type":$pointeeType,
            "int":$addressSpace
        ), [{
            return $_get(pointeeType.getContext(), pointeeType, addressSpace);
        }]>
    ];

    let skipDefaultBuilders = 1;
}
```
In MLIR TableGen, the `let builders = [...]` section allows you to define custom C++ factory methods for creating instances of your Type or Attribute. In the generated C++ code, these become static `get(...)` methods on your type's class.

Here is a straightforward breakdown of exactly what is happening in the `let builders` block of your Triton pointer type snippet.

### Syntax Breakdown

```tablegen
let builders = [
    TypeBuilderWithInferredContext<(ins
        "Type":$pointeeType,
        "int":$addressSpace
    ), [{
        return $_get(pointeeType.getContext(), pointeeType, addressSpace);
    }]>
];
```

* **`let builders = [...]`**: This tells TableGen, "Generate these specific factory methods in the C++ class." You can provide a list of multiple builders if you want different ways to construct the type.
* **`TypeBuilderWithInferredContext`**: Every Type in MLIR must be registered in an `MLIRContext`. Usually, this means the very first argument to any `get()` method must be `MLIRContext *context`. However, this specific TableGen class tells the compiler: *"Don't force the user to pass the context explicitly. We can figure it out from the other arguments."*
* **`(ins "Type":$pointeeType, "int":$addressSpace)`**: This defines the signature of your C++ builder function. It will take two arguments: an MLIR `Type` (the type being pointed to) and an `int` (the memory address space). Notice that `MLIRContext *` is absent.
* **`[{ return $_get(...); }]`**: The `[{ ... }]` syntax denotes a block of raw C++ code. This is the actual implementation of your builder function.
* **`$_get`**: This is a special TableGen substitution variable. When TableGen generates the C++, it replaces `$_get` with a call to the underlying auto-generated method that handles memory allocation and uniquing (typically `Base::get(...)`). 
* **`pointeeType.getContext()`**: Because MLIR types always carry a pointer to their context, the builder reaches into the `pointeeType` parameter to extract the `MLIRContext` and passes it to `$_get`. This is how the context is "inferred."

### Context: How does this tie into the rest of the snippet?

If you look at the bottom of your snippet, you will see this line:
```tablegen
let skipDefaultBuilders = 1;
```
By default, because you defined `parameters = (ins "Type":$pointeeType, "int":$addressSpace)`, TableGen wants to automatically generate a standard builder that looks like this:
```cpp
static PointerType get(MLIRContext *context, Type pointeeType, int addressSpace);
```
By providing your own custom builder and setting `skipDefaultBuilders = 1`, you suppress the default one. 

### What does this generate in C++?

When you run `mlir-tblgen`, this specific `builders` block will generate a highly convenient static C++ method inside the `triton::PointerType` class that looks exactly like this:

```cpp
// The generated C++ factory method
static PointerType get(Type pointeeType, int addressSpace) {
    // It extracts the context from pointeeType automatically
    return Base::get(pointeeType.getContext(), pointeeType, addressSpace);
}
```

**Why do this?** It's purely for developer ergonomics. It makes the C++ API much cleaner to use. Instead of writing `PointerType::get(myType.getContext(), myType, 1)`, a C++ developer writing compiler passes can just write `PointerType::get(myType, 1)`.


# `assemblyFormat` vs `hasCustomAssemblyFormat`
* **`let assemblyFormat = "...";`**: TableGen writes the `TypeSwitch`, *and* it writes the `print`/`parse` methods for you automatically.
* **`let hasCustomAssemblyFormat = 1;`**: TableGen writes the `TypeSwitch`, but leaves the `print`/`parse` methods blank for you to implement manually in C++. This is useful when your attribute has highly complex syntax that the declarative `assemblyFormat` string can't handle.
