# mlir::dyn_cast_or_null<T>(val)
**`mlir::dyn_cast<T>(val)` requires a non-null value, while `mlir::dyn_cast_or_null<T>(val)` safely accepts null.**

### Behavior details

Both are part of the LLVM-style casting utilities (used heavily in MLIR for `Type`, `Attribute`, `Value`, `Operation*`, concrete ops, etc.). They attempt a dynamic cast and return a null/empty result of type `T` on failure (instead of asserting like `cast<T>`).

- **`mlir::dyn_cast<T>(val)`**  
  - Assumes `val` is **non-null**.  
  - Internally calls something equivalent to `isa<T>(val)` first.  
  - `isa` asserts in debug builds if the value is null (`assert(impl && "isa<> used on a null type.")` or similar).  
  - Returns a null/empty `T` only if the type check fails.

- **`mlir::dyn_cast_or_null<T>(val)`** (also known as the older name for what is now often `dyn_cast_if_present`)  
  - Explicitly handles null.  
  - First checks whether the value is present/non-null.  
  - If null → immediately returns a null/empty `T` (no assert).  
  - If non-null → behaves exactly like `dyn_cast`.

### Classic implementation pattern (from MLIR `Type` / similar classes)

```cpp
template <typename U>
U dyn_cast() const {
  return isa<U>() ? U(impl) : U(nullptr);   // asserts if null
}

template <typename U>
U dyn_cast_or_null() const {
  return (impl && isa<U>()) ? U(impl) : U(nullptr);  // safe on null
}
```

### Practical usage

```cpp
// Safe when the defining op might be null
if (auto op = mlir::dyn_cast_or_null<MyOp>(value.getDefiningOp())) {
  // ...
}

// Prefer this only when you know the pointer/value is non-null
if (auto op = mlir::dyn_cast<MyOp>(someDefinitelyNonNullOp)) {
  // ...
}
```

**Modern note**: Prefer the free functions (`mlir::dyn_cast` / `mlir::dyn_cast_or_null` or the `llvm::` equivalents) over the old member functions, which have been deprecated. In recent LLVM, `dyn_cast_or_null` forwards to the more general `dyn_cast_if_present`.