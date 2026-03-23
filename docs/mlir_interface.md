<!-- TOC START -->
- [MLIR Interface](#mlir-interface)
  - [How `iface` is created](#how-iface-is-created)
  - [Concept-based Polymorphism](#concept-based-polymorphism)
    - [The Problem with Traditional Polymorphism](#the-problem-with-traditional-polymorphism)
    - [The Solution: Concept-Based Polymorphism](#the-solution-concept-based-polymorphism)
      - [1. The Concept (The Contract)](#1-the-concept-the-contract)
      - [2. The Model (The Bridge)](#2-the-model-the-bridge)
      - [3. The Wrapper (The Type Eraser)](#3-the-wrapper-the-type-eraser)
    - [The Ultimate Superpower: Retroactive Interfaces](#the-ultimate-superpower-retroactive-interfaces)
  - [The connection between a CustomOp and ExampleOpInterface](#the-connection-between-a-customop-and-exampleopinterface)
    - [1. The Setup: The Context's Secret Dictionary](#1-the-setup-the-contexts-secret-dictionary)
    - [2. The Connection: `dyn_cast`](#2-the-connection-dyn_cast)
    - [Summary of the Flow](#summary-of-the-flow)
  - [Model<CustomOp> plays the role of a vtable](#modelcustomop-plays-the-role-of-a-vtable)
    - [Traditional C++ (The Bloat)](#traditional-c-the-bloat)
    - [MLIR's Approach (The Solution)](#mlirs-approach-the-solution)
<!-- TOC END -->


# MLIR Interface
The client code:
```C++
    if (auto iface = dyn_cast<example::ExampleOpInterface>(op)) {
      llvm::outs() << op->getName() << " magic number: " 
                   << iface.getMagicNumber() << "\n";
    } else {
      llvm::outs() << op->getName() << " does not implement ExampleOpInterface.\n";
    }
```
iface:<br/>
<img src="./images/iface.png">
<br/>

## How `iface` is created
**Call Stack:**
```C++
mlir::detail::Interface<mlir::example::ExampleOpInterface, mlir::Operation*, mlir::example::detail::ExampleOpInterfaceInterfaceTraits, mlir::Op<mlir::example::ExampleOpInterface>, mlir::OpTrait::TraitBase>::Interface(mlir::Operation*) (/data00/home/son.nguyen/workspace/learnmlir/llvm-project/mlir/include/mlir/Support/InterfaceSupport.h:95)
mlir::OpInterface<mlir::example::ExampleOpInterface, mlir::example::detail::ExampleOpInterfaceInterfaceTraits>::OpInterface(mlir::Operation*) (/data00/home/son.nguyen/workspace/learnmlir/llvm-project/mlir/include/mlir/IR/OpDefinition.h:2094)
mlir::example::ExampleOpInterface::ExampleOpInterface(mlir::Operation*) (/data00/home/son.nguyen/workspace/learnmlir/examples/mlir_interface/build/ExampleInterface.h.inc:53)
llvm::ValueFromPointerCast<mlir::example::ExampleOpInterface, mlir::Operation, llvm::CastInfo<mlir::example::ExampleOpInterface, mlir::Operation*, void>>::doCast(mlir::Operation*) (/data00/home/son.nguyen/workspace/learnmlir/llvm-project/llvm/include/llvm/Support/Casting.h:335)
llvm::DefaultDoCastIfPossible<mlir::example::ExampleOpInterface, mlir::Operation*, llvm::CastInfo<mlir::example::ExampleOpInterface, mlir::Operation*, void>>::doCastIfPossible(mlir::Operation*) (/data00/home/son.nguyen/workspace/learnmlir/llvm-project/llvm/include/llvm/Support/Casting.h:313)
decltype(auto) llvm::dyn_cast<mlir::example::ExampleOpInterface, mlir::Operation>(mlir::Operation*) (/data00/home/son.nguyen/workspace/learnmlir/llvm-project/llvm/include/llvm/Support/Casting.h:663)
main::$_0::operator()(mlir::Operation*) const (/data00/home/son.nguyen/workspace/learnmlir/examples/mlir_interface/main.cpp:36)
main (/data00/home/son.nguyen/workspace/learnmlir/examples/mlir_interface/main.cpp:46)
__libc_start_call_main (libc_start_call_main.h:58)
__libc_start_main_impl (libc-start.c:360)
_start (:12)
```
<br/>

The following inherited ctor `Interface(ValueT t = ValueT())` is called:<br/>
llvm-project/mlir/include/mlir/Support/InterfaceSupport.h
```C++
template <typename ConcreteType, typename ValueT, typename Traits,
          typename BaseType,
          template <typename, template <typename> class> class BaseTrait>
class Interface : public BaseType {
public:
  using Concept = typename Traits::Concept;
  template <typename T>
  using Model = typename Traits::template Model<T>;
  template <typename T>
  using FallbackModel = typename Traits::template FallbackModel<T>;
  using InterfaceBase =
      Interface<ConcreteType, ValueT, Traits, BaseType, BaseTrait>;
  template <typename T, typename U>
  using ExternalModel = typename Traits::template ExternalModel<T, U>;
  using ValueType = ValueT;

  /// Construct an interface from an instance of the value type.
  explicit Interface(ValueT t = ValueT())
      : BaseType(t),
        conceptImpl(t ? ConcreteType::getInterfaceFor(t) : nullptr) {
    assert((!t || conceptImpl) &&
           "expected value to provide interface instance");
  }

}

// ValueT = Operation*
// ConcreteType = ExampleOpInterface
// BaseType = mlir::Op<mlir::example::ExampleOpInterface>
```

The `Interface(ValueT t = ValueT())` creates a `mlir::Op<mlir::example::ExampleOpInterface>` instance from the `op` argument.<br/>
Since `ExampleOpInterface::getInterfaceFor(op)` returns a pointer to `detail::ExampleOpInterfaceInterfaceTraits::Model<CustomOp>`, `conceptImpl` is initialized to this pointer.
<b/r>

## Concept-based Polymorphism
Concept-based polymorphism (often referred to interchangeably with **type erasure**) is a powerful C++ design pattern. It allows you to write generic code that handles different types uniformly, *without* forcing those types to inherit from a common base class or use virtual functions.

To understand why MLIR uses this, it helps to look at the problem it solves.

### The Problem with Traditional Polymorphism
In standard object-oriented C++, if you want a generic `getMagicNumber()` function, you create a base class with a `virtual` method:

```cpp
class MagicInterface {
public:
  virtual int getMagicNumber() = 0;
};

class CustomOp : public MagicInterface { ... };
```

This has two major drawbacks for a compiler infrastructure like MLIR:
1. **Memory Bloat (The VTable Pointer):** Every single operation instance in memory would need a hidden pointer (the vptr) to resolve virtual function calls. In a compiler with millions of IR nodes, this wastes a massive amount of memory and ruins cache locality.
2. **Intrusive Inheritance:** You can only use the interface if you modify the original class to inherit from `MagicInterface`. You cannot retroactively make a third-party class implement your interface.

### The Solution: Concept-Based Polymorphism
Concept-based polymorphism fixes this by moving the inheritance and virtual dispatch *out* of the object itself and into a hidden side-channel. 

It relies on three main components. Here is how they map directly to the MLIR code you generated earlier:

#### 1. The Concept (The Contract)
The Concept defines what methods the interface requires. However, instead of the concrete operation inheriting from it, it exists purely as an abstract definition. 

In MLIR's generated code, this is a struct of function pointers:
```cpp
struct Concept {
  int (*getMagicNumber)(const Concept *impl, ::mlir::Operation *);
};
```
*Notice:* `CustomOp` knows absolutely nothing about this struct.

#### 2. The Model (The Bridge)
The Model is a templated class that "bridges" the generic Concept to a specific, concrete type. It knows how to translate the interface's demands into the specific methods of the concrete class.

In your MLIR code, the framework generates a `Model` for every operation that claims to implement the interface:
```cpp
template<typename ConcreteOp>
class Model : public Concept {
public:
  Model() : Concept{getMagicNumber} {} // Wire up the function pointer

  static inline int getMagicNumber(const Concept *impl, ::mlir::Operation *op) {
    // Safely cast the opaque pointer back to the real type, 
    // and call its normal, non-virtual C++ method.
    return (llvm::cast<ConcreteOp>(op)).getMagicNumber();
  }
};
```
When `CustomOp` is registered, MLIR instantiates `Model<CustomOp>` once and stores it in a registry inside the `MLIRContext`.

#### 3. The Wrapper (The Type Eraser)
This is the class you actually interact with. It acts as a lightweight proxy. It holds two things: a pointer to the actual data (the operation), and a pointer to the Concept (the behavior).

In your code, this is the `ExampleOpInterface` class:
```cpp
class ExampleOpInterface {
  Operation *op;         // The opaque data
  const Concept *concept; // The behavior dictionary

public:
  int getMagicNumber() {
    // Delegate the call through the concept's function pointer
    return concept->getMagicNumber(concept, op);
  }
};
```

### The Ultimate Superpower: Retroactive Interfaces
Because the operation (`CustomOp`) and the interface mechanism (`Model` and `Concept`) are completely decoupled, you gain a massive advantage: **you can write interfaces for types you don't own.**

For example, imagine you want the standard `arith.addi` operation (built into MLIR) to implement your `ExampleOpInterface`. You cannot modify the `arith` dialect's source code. 

With concept-based polymorphism, you don't have to. You can use the **`ExternalModel`** (which was generated in your C++ snippet). You simply define how `arith.addi` should behave, instantiate an `ExternalModel<MyArithModel, arith::AddIOp>`, and register it with the `MLIRContext`. From then on, any `dyn_cast<ExampleOpInterface>` on an `arith.addi` operation will succeed.


## The connection between a CustomOp and ExampleOpInterface
The connection happens dynamically inside the **MLIR infrastructure**—specifically, when you attempt to cast the operation using `dyn_cast`. 

The `MLIRContext` acts as a "matchmaker" between your `CustomOp` data and the `Concept` behavior. Here is exactly how that connection is made:

### 1. The Setup: The Context's Secret Dictionary
When you load your dialect into the `MLIRContext` (via `context.getOrLoadDialect<ExampleDialect>()`), MLIR does some hidden bookkeeping. 

Because `CustomOp` was defined with the `ExampleOpInterface::Trait`, the MLIR framework automatically instantiates the `Model<CustomOp>` (which, remember, is just a subclass of `Concept`). It then stores a pointer to this `Concept` in a massive dictionary inside the `MLIRContext`.

This dictionary maps the unique **TypeID** of an operation to its registered **Concept pointers**:
* `TypeID(CustomOp)` ➔ `Concept*` (specifically, the `Model<CustomOp>` instance)
* `TypeID(DefaultOp)` ➔ `Concept*` (specifically, the `Model<DefaultOp>` instance)



Notice that `CustomOp` instances themselves *do not* store the `Concept*`. They only store their data and their `TypeID`. 

### 2. The Connection: `dyn_cast`
The actual wiring of the `op` and `concept` data members happens the moment you call `dyn_cast`.

When you write this line of code:
```cpp
Operation *rawOp = customOp.getOperation();
auto iface = dyn_cast<ExampleOpInterface>(rawOp);
```

Here is what MLIR's `dyn_cast` is doing under the hood to populate those two fields:

1. **Find the Data (`op`):** It already has the raw `Operation*` pointer (your `rawOp`), so that becomes the `op` member.
2. **Find the Behavior (`concept`):** * It looks at `rawOp` and asks, "What is your TypeID?" (Answer: `CustomOp`'s TypeID).
   * It goes to the `MLIRContext`'s dictionary and looks up `CustomOp`'s TypeID.
   * The dictionary returns the `Concept*` that was registered during setup.
3. **Assemble the Wrapper:** `dyn_cast` now has both pieces. It calls a hidden constructor on `ExampleOpInterface` that essentially looks like this:
   ```cpp
   // Pseudocode of what dyn_cast does internally
   ExampleOpInterface(Operation *op, const Concept *concept) {
       this->op = op;
       this->concept = concept;
   }
   ```

### Summary of the Flow
* **`CustomOp`** provides the pure data (and its TypeID).
* **`MLIRContext`** holds the behavior dictionary (`Concept` pointers).
* **`dyn_cast`** takes the data, looks up the correct behavior in the dictionary using the TypeID, and bundles them both together into the `ExampleOpInterface` wrapper.

Once `dyn_cast` returns that wrapper, you have an object that contains both the specific `Operation*` and the specific `Concept*` needed to execute `iface.getMagicNumber()`. 


## Model<CustomOp> plays the role of a vtable
This is the exact core design philosophy behind MLIR's interface system. **`Model<ConcreteOp>` is exactly a vtable**, but instead of storing a pointer to it inside every single object, MLIR stores it globally in the `MLIRContext`.

Here is a quick breakdown of why this completely solves the Memory Bloat problem:



### Traditional C++ (The Bloat)
In standard C++, if `CustomOp` had virtual functions, the compiler would silently inject a **vptr** (virtual pointer, usually 8 bytes) into *every single instance* of `CustomOp` you create. 
* If you compile a program that generates 1,000,000 `CustomOp` operations, you instantly consume **8 Megabytes** of memory just storing duplicate `vptr`s that all point to the exact same vtable.
* Furthermore, these pointers ruin cache locality because the CPU has to dereference the pointer inside the object to find the function it needs to call.

### MLIR's Approach (The Solution)
In MLIR, the `Operation` class is heavily optimized. It contains only what is strictly necessary (pointers to its operands, results, attributes, and its `OperationName`/`TypeID`). **There is no vptr.**

Instead of putting the vptr in the object:
1. MLIR creates **exactly one** instance of `Model<CustomOp>` (your vtable) when the dialect is initialized.
2. It stores this single instance inside a hash map inside the `MLIRContext`, keyed by the operation's `TypeID`.
3. If you create 1,000,000 `CustomOp` instances, the memory overhead for the interface is **0 bytes**. 

The tradeoff is that when you call `dyn_cast<ExampleOpInterface>(op)`, MLIR has to do a quick hash map lookup in the `MLIRContext` to find the `Model` rather than just dereferencing a pointer. However, because compiler transformations usually cast once and then call interface methods many times, this tradeoff heavily favors MLIR's design.

