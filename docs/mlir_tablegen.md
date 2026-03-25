<!-- TOC START -->
- [Parameters](#parameters)
- [Builders](#builders)
    - [Syntax Breakdown](#syntax-breakdown)
    - [Context: How does this tie into the rest of the snippet?](#context-how-does-this-tie-into-the-rest-of-the-snippet)
    - [What does this generate in C++?](#what-does-this-generate-in-c)
- [`assemblyFormat` vs `hasCustomAssemblyFormat`](#assemblyformat-vs-hascustomassemblyformat)
<!-- TOC END -->


# Dialect useDefaultAttributePrinterParser
```MLIR
def Example_Dialect : Dialect {
  let name = "example";
  let cppNamespace = "::mlir::example";
  let summary = "An example dialect for demonstrating interfaces.";

  // CRUCIAL: Tells the dialect to use the auto-generated dispatch hooks
  // for all attributes that define a `mnemonic` and (`assemblyFormat` or `hasCustomAssemblyFormat`).
  let useDefaultAttributePrinterParser = 1;
}
```

# Dialect useDefaultTypePrinterParser
In MLIR TableGen, the `let useDefaultTypePrinterParser = 1;` syntax does something incredibly useful for parsing and printing, but there is an important distinction to make here: unlike `parameters`, `builders`, or `genVerifyDecl`, this flag is typically set on the **Dialect** definition itself, not on the individual types.

Here is a straightforward breakdown of what this flag means and how it eliminates a massive amount of C++ boilerplate.

### Syntax Breakdown

* **`let useDefaultTypePrinterParser = 1;`**: You are setting a boolean flag to `1` (true) inside your dialect's TableGen class (e.g., `def MyDialect : Dialect { ... }`). 

### Context: The Problem It Solves

When MLIR reads a `.mlir` text file, it encounters textual representations of types (like `!triton.ptr<f32, 1>`). The MLIR parser sees the `!triton.` prefix and hands the rest of the string over to the Triton dialect, essentially saying: *"Hey Triton, parse this type for me."*

Similarly, when MLIR prints IR back to text, it asks the dialect to print its types. 

Historically, to handle this, you had to manually write giant `switch` statements in your dialect's C++ implementation file to figure out which specific type was being parsed or printed. It looked something like this:

```cpp
// The old, manual way (without the flag)
::mlir::Type TritonDialect::parseType(::mlir::DialectAsmParser &parser) const {
    llvm::StringRef mnemonic;
    if (parser.parseKeyword(&mnemonic)) return Type();
    
    if (mnemonic == "ptr") return PointerType::parse(parser);
    if (mnemonic == "other_type") return OtherType::parse(parser);
    // ... endless boilerplate for every type ...
    return Type();
}
```
Every time you added a new type to your dialect, you had to remember to update this C++ method.

### What does this generate in C++?

By setting `let useDefaultTypePrinterParser = 1;` in your Dialect's `.td` file, you tell TableGen to automatically generate that giant `switch` statement for you. 

TableGen looks at every `TypeDef` registered to your dialect, extracts their `mnemonic` strings (like `"ptr"` from your Triton pointer example), and automatically implements the following methods in your Dialect's C++ class:

1.  **`::mlir::Type MyDialect::parseType(::mlir::DialectAsmParser &parser) const;`**
2.  **`void MyDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const;`**

When the generated `parseType` method matches the mnemonic `"ptr"`, it automatically dispatches the rest of the parsing work to your `PointerType` class. 

### How does this connect to your individual Types?

For this auto-generated dialect parser to actually work, your individual types must know how to parse and print themselves. They do this in one of two ways:

1.  **Declaratively:** By defining an `assemblyFormat = "...";` in the type's `.td` file.
2.  **Manually in C++:** By setting `hasCustomAssemblyFormat = 1;` in the type's `.td` file (as seen in your Triton pointer snippet) and manually writing the `parse` and `print` C++ methods.


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

TableGen will also generate a class for this type which has a `get(Type pointeeType, int addressSpace)` method:
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

**Why do this?** It's purely for developer ergonomics. It makes the C++ API much cleaner to use. Instead of writing `PointerType::get(myType.getContext(), myType, 1)`, a C++ developer writing compiler passes can just write `PointerType::get(myType, 1)`.<br/>


# assemblyFormat vs hasCustomAssemblyFormat
* **`let assemblyFormat = "...";`**: TableGen writes the `TypeSwitch`, *and* it writes the `print`/`parse` methods for you automatically.
* **`let hasCustomAssemblyFormat = 1;`**: TableGen writes the `TypeSwitch`, but leaves the `print`/`parse` methods blank for you to implement manually in C++. This is useful when your attribute has highly complex syntax that the declarative `assemblyFormat` string can't handle.


# Verify
In MLIR TableGen, the `let genVerifyDecl = 1;` syntax tells the compiler to generate a C++ **declaration** for a verification method, but leaves the actual **implementation** up to you to write in your C++ source file.

Here is a straightforward breakdown of what this means and why you need it for custom Types or Attributes.

### Syntax Breakdown

* **`let genVerifyDecl = 1;`**: You are setting the boolean flag `genVerifyDecl` (generate verification declaration) to `1` (true). By default, this is usually `0` (false) for Types and Attributes, meaning no custom verification is required.

### Context: Why do we need verification?

When you define `parameters` for a Type (like the `pointeeType` and `addressSpace` in your Triton pointer example), TableGen knows their C++ types (`Type` and `int`). However, TableGen does not know your **domain-specific rules**. 

For example:
* What if a developer tries to create a `PointerType` where the `pointeeType` is a `FunctionType` or a `VoidType`, but your compiler only allows pointers to scalars or tensors?
* What if they pass a negative number for `addressSpace`, which might be invalid for your hardware target?

MLIR creates Types and Attributes via a uniquing system (the `get()` methods). Before MLIR allocates memory for a new Type, it needs a way to check if the provided parameters are semantically valid. Setting `genVerifyDecl = 1` provides the hook to enforce these rules.

### What does this generate in C++?

When you run `mlir-tblgen`, this flag forces the generator to add a static `verify` method signature to your generated C++ class header (`.h.inc` file). 

Using the Triton pointer type as an example, TableGen will generate a declaration that looks like this:

```cpp
// Auto-generated inside the PointerType class definition
static ::mlir::LogicalResult verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    Type pointeeType, 
    int addressSpace
);
```
* **`emitError`**: A callback function provided by MLIR. If the parameters are invalid, you use this to report exactly what went wrong.
* **The parameters**: It passes in the exact parameters you defined in `let parameters = ...` so you can inspect them.
* **`LogicalResult`**: The method must return `success()` if the parameters are valid, or `failure()` if they are invalid.

### What do YOU have to write?

Because you told TableGen to only generate the *declaration*, the C++ compiler will throw a linker error until you write the actual implementation in your `.cpp` file. 

You would manually write something like this:

```cpp
::mlir::LogicalResult PointerType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    Type pointeeType, 
    int addressSpace) {
    
    // Rule 1: Address space cannot be negative
    if (addressSpace < 0) {
        return emitError() << "address space cannot be negative";
    }

    // Rule 2: Cannot point to a function type
    if (isa<FunctionType>(pointeeType)) {
        return emitError() << "pointers to functions are not supported";
    }

    return ::mlir::success();
}
```

By doing this, any time `PointerType::get(...)` is called anywhere in your compiler, MLIR will automatically call your `verify` function first. If `verify` fails, MLIR will gracefully emit an error and abort the type creation, preventing your compiler from crashing later on due to malformed types.
