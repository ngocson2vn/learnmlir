# IntegerType TypeID

## MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::IntegerType)
llvm-project/build/tools/mlir/include/mlir/IR/BuiltinTypes.h.inc
```C++
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::IntegerType)
```

llvm-project/mlir/include/mlir/Support/TypeID.h
```C++
#define MLIR_DECLARE_EXPLICIT_TYPE_ID(CLASS_NAME)                              \
  MLIR_DECLARE_EXPLICIT_SELF_OWNING_TYPE_ID(CLASS_NAME)

// Declare/define an explicit specialization for TypeID: this forces the
// compiler to emit a strong definition for a class and controls which
// translation unit and shared object will actually have it.
// This can be useful to turn to a link-time failure what would be in other
// circumstances a hard-to-catch runtime bug when a TypeID is hidden in two
// different shared libraries and instances of the same class only gets the same
// TypeID inside a given DSO.
#define MLIR_DECLARE_EXPLICIT_SELF_OWNING_TYPE_ID(CLASS_NAME)                  \
  namespace mlir {                                                             \
  namespace detail {                                                           \
  template <>                                                                  \
  class TypeIDResolver<CLASS_NAME> {                                           \
  public:                                                                      \
    static TypeID resolveTypeID() { return id; }                               \
                                                                               \
  private:                                                                     \
    static SelfOwningTypeID id;                                                \
  };                                                                           \
  } /* namespace detail */                                                     \
  } /* namespace mlir */


//===----------------------------------------------------------------------===//
// SelfOwningTypeID
//===----------------------------------------------------------------------===//

/// Defines a TypeID for each instance of this class by using a pointer to the
/// instance. Thus, the copy and move constructor are deleted.
/// Note: We align by 8 to match the alignment of TypeID::Storage, as we treat
/// an instance of this class similarly to TypeID::Storage.
class alignas(8) SelfOwningTypeID {
public:
  SelfOwningTypeID() = default;
  SelfOwningTypeID(const SelfOwningTypeID &) = delete;
  SelfOwningTypeID &operator=(const SelfOwningTypeID &) = delete;
  SelfOwningTypeID(SelfOwningTypeID &&) = delete;
  SelfOwningTypeID &operator=(SelfOwningTypeID &&) = delete;

  /// Implicitly converts to the owned TypeID.
  operator TypeID() const { return getTypeID(); }

  /// Return the TypeID owned by this object.
  TypeID getTypeID() const { return TypeID::getFromOpaquePointer(this); }
};
```

## MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::IntegerType)
llvm-project/build/tools/mlir/include/mlir/IR/BuiltinTypes.cpp.inc
```C++
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::IntegerType)
```

llvm-project/mlir/include/mlir/Support/TypeID.h
```C++
#define MLIR_DEFINE_EXPLICIT_TYPE_ID(CLASS_NAME)                               \
  MLIR_DEFINE_EXPLICIT_SELF_OWNING_TYPE_ID(CLASS_NAME)

#define MLIR_DEFINE_EXPLICIT_SELF_OWNING_TYPE_ID(CLASS_NAME)                   \
  namespace mlir {                                                             \
  namespace detail {                                                           \
  SelfOwningTypeID TypeIDResolver<CLASS_NAME>::id = {};                        \
  } /* namespace detail */                                                     \
  } /* namespace mlir */
```

## class TypeID
llvm-project/mlir/include/mlir/Support/TypeID.h

## Get TypeID
Client code:
```C++
auto typeID = mlir::TypeID::get<mlir::IntegerType>();
llvm::outs() << "typeID: " << typeID.getAsOpaquePointer() << "\n";

// llvm-project/mlir/include/mlir/Support/TypeID.h
template <typename T>
TypeID TypeID::get() {
  return detail::TypeIDResolver<T>::resolveTypeID();
}

// detail::TypeIDResolver<T>::resolveTypeID() returns the static SelfOwningTypeID id which is then implicitly converted to TypeID
```

