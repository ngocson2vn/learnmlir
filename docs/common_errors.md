# Common Errors
## 1. operand #0 does not dominate this use
Solution:  
Modify llvm-project/mlir/lib/IR/Verifier.cpp to print out the faulty op:
```C++
/// Emit an error when the specified operand of the specified operation is an
/// invalid use because of dominance properties.
static void diagnoseInvalidOperandDominance(Operation &op, unsigned operandNo) {
  InFlightDiagnostic diag = op.emitError("operand #")
                            << operandNo << " does not dominate this use: "
                            << op;
```

## 2. cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: builtin.module
https://medium.com/@ngocson2vn/mlir-error-cannot-be-converted-to-llvm-ir-missing-llvmtranslationdialectinterface-registration-95a66e86b8c1

Solution:
```C++
mlir::DialectRegistry registry;

// This registration is required for translating builtin.module op
// Otherwise, MLIR will report "error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: builtin.module"
mlir::registerBuiltinDialectTranslation(registry);

// This registration is required for translating LLVM dialect ops
mlir::registerLLVMDialectTranslation(registry);

// Set up the MLIR context
MLIRContext context(registry);

//
// Build moduleOp
//
OpBuilder builder(&context);
...

// Lower to LLVM module.
llvm::LLVMContext llvmContext;
auto llvmModule = mlir::translateModuleToLLVMIR(moduleOp, llvmContext);
```

## 3. undefined reference to `typeinfo for mlir::Dialect'
### The Root Cause: RTTI Mismatch

The root cause of the `undefined reference to 'typeinfo for mlir::Dialect'` error is a **Runtime Type Information (RTTI) mismatch** between your application and the MLIR/LLVM libraries you are linking against.

Here is exactly what is happening under the hood:
1. **LLVM/MLIR Defaults:** By default, LLVM and MLIR are built with C++ RTTI disabled (using the `-fno-rtti` compiler flag) to save memory and improve performance. They use their own highly optimized casting system (e.g., `llvm::cast`, `llvm::dyn_cast`) instead of standard C++ `dynamic_cast`.
2. **Your Application:** Your application is likely being compiled with RTTI enabled (which is the default behavior for standard C++ compilers like GCC and Clang).
3. **The Clash:** Because `mlir::Dialect` has virtual functions, your compiler expects to find standard C++ `typeinfo` generated for it so that things like `dynamic_cast` can work. However, since the MLIR libraries were built *without* RTTI, that `typeinfo` simply doesn't exist in the `.a` or `.so` files you are linking against, resulting in the undefined reference.

---


### The relationship between `typeinfo` and the linker

#### 1. What is `typeinfo`? (The Compiler's Job)
In standard C++, when a class has at least one virtual function (making it a polymorphic class), the compiler generates extra metadata for it. This metadata allows the program to figure out the exact type of an object at runtime. 

This system is called **RTTI (Runtime Type Information)**. RTTI relies on a hidden, compiler-generated data structure for each polymorphic class, known as the `typeinfo` object. 
* **What uses it:** C++ uses the `typeinfo` struct to power features like `dynamic_cast` and the `typeid` operator.
* **Where it lives:** The compiler typically emits the `typeinfo` object into the compiled object file (`.o`) alongside the class's Virtual Method Table (vtable). 

#### 2. What is the Linker's Job?
The linker acts as a matchmaker. When your compiler finishes translating your `.cpp` files into object files (`.o`), those object files are full of "holes" or **unresolved symbols**. 

For example, if you call `printf`, your object file doesn't contain the code for `printf`. It just contains a symbolic reference saying: *"I need the function `printf`."* The linker's job is to search through all the other object files and libraries (`.a` or `.so` files) you provided, find the actual definition of `printf`, and wire them together. If it cannot find the definition anywhere, it throws an `undefined reference` error.

#### 3. The Collision: Why the Error Happens

Now, let's look at exactly what happens when you compile your MLIR application with RTTI enabled, but link against MLIR libraries built with RTTI disabled (`-fno-rtti`).

**Step A: Your code makes a promise (The Reference)**
When you write an MLIR application, you are almost always subclassing MLIR base classes. For example, you might create `class MyDialect : public mlir::Dialect`. 
Because your application is compiled with RTTI enabled, the compiler tries to generate a standard C++ `typeinfo` object for `MyDialect`. However, because `MyDialect` inherits from `mlir::Dialect`, the `typeinfo` for `MyDialect` must legally contain a reference to the `typeinfo` of its base class (`mlir::Dialect`). 

Therefore, the compiler inserts an unresolved symbol into your application's object file: *"Hey Linker, I am going to need the `typeinfo for mlir::Dialect`."*

**Step B: The library breaks the promise (The Missing Definition)**
When the LLVM/MLIR developers compiled the `libMLIR.a` or `libMLIR.so` libraries, they used the `-fno-rtti` flag. This explicitly told their compiler: *"Do not generate `typeinfo` structs for any classes, including `mlir::Dialect`."* Therefore, the `typeinfo` structure simply does not exist inside the compiled MLIR binaries.

**Step C: The Linker Fails**
The linker takes over. It sees that your application needs `typeinfo for mlir::Dialect`. It searches exhaustively through all the MLIR libraries you linked against. It finds the methods for `mlir::Dialect`, it finds the variables, but because of `-fno-rtti`, it cannot find the `typeinfo` symbol. 

Since the linker cannot fulfill your object file's request, it throws up its hands and reports: **`undefined reference to 'typeinfo for mlir::Dialect'`**.

---

By adding `-fno-rtti` to your own application's build flags, you instruct your compiler to stop generating `typeinfo` for your classes, which stops it from asking the linker to find the `typeinfo` for MLIR's base classes.


### The Fix: Disable RTTI in Your Project

To resolve this, you need to compile your application with RTTI disabled so that it matches how the MLIR libraries were built. 

**If you are using CMake (Recommended)**
Add the `-fno-rtti` flag to your CXX flags. You can do this by adding the following line to your `CMakeLists.txt` before you compile your targets:

```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
```