# MLIR Builtin Types
When the client code creates a `mlir::MLIRContext context` object, the following process will happen: <br/>

## Summary
```C++
MLIRContext::MLIRContext(Threading setting)
  -> MLIRContextImpl(bool threadingIsEnabled)
    -> getOrLoadDialect<BuiltinDialect>()
      -> BuiltinDialect::BuiltinDialect(::mlir::MLIRContext *context)
        -> BuiltinDialect::initialize()
          -> BuiltinDialect::registerTypes()
            -> Dialect::addTypes<BuiltinTypes...>()
               BuiltinTypes are defined in `mlir/IR/BuiltinTypes.cpp.inc`
```

## Step 1: MLIRContext loads BuiltinDialect
MLIRContext -> new MLIRContextImpl(...) -> getOrLoadDialect<BuiltinDialect>()
```C++
// llvm-project/mlir/lib/IR/MLIRContext.cpp
class MLIRContextImpl {
public:
  DenseMap<TypeID, AbstractType *> registeredTypes;
  StorageUniquer typeUniquer;

  /// This is a mapping from type name to the abstract type describing it.
  /// It is used by `AbstractType::lookup` to get an `AbstractType` from a name.
  /// As this map needs to be populated before `StringAttr` is loaded, we
  /// cannot use `StringAttr` as the key. The context does not take ownership
  /// of the key, so the `StringRef` must outlive the context.
  llvm::DenseMap<StringRef, AbstractType *> nameToType;

  /// This is a list of dialects that are created referring to this context.
  /// The MLIRContext owns the objects. These need to be declared after the
  /// registered operations to ensure correct destruction order.
  DenseMap<StringRef, std::unique_ptr<Dialect>> loadedDialects;

  // ...
}

MLIRContext::MLIRContext(Threading setting)
    : MLIRContext(DialectRegistry(), setting) {}

MLIRContext::MLIRContext(const DialectRegistry &registry, Threading setting)
    : impl(new MLIRContextImpl(setting == Threading::ENABLED &&
                               !isThreadingGloballyDisabled())) {
  // Initialize values based on the command line flags if they were provided.
  if (clOptions.isConstructed()) {
    printOpOnDiagnostic(clOptions->printOpOnDiagnostic);
    printStackTraceOnDiagnostic(clOptions->printStackTraceOnDiagnostic);
  }

  // Pre-populate the registry.
  registry.appendTo(impl->dialectsRegistry);

  // Ensure the builtin dialect is always pre-loaded.
  getOrLoadDialect<BuiltinDialect>();

  // ...
}
```

## Step 2: Create a BuiltinDialect instance
```C++
// llvm-project/mlir/include/mlir/IR/MLIRContext.h
  /// Get (or create) a dialect for the given derived dialect type. The derived
  /// type must provide a static 'getDialectNamespace' method.
  template <typename T>
  T *getOrLoadDialect() {
    return static_cast<T *>(
        getOrLoadDialect(T::getDialectNamespace(), TypeID::get<T>(), [this]() {
          std::unique_ptr<T> dialect(new T(this));
          return dialect;
        }));
  }

// T = BuiltinDialect
// llvm-project/mlir/lib/IR/MLIRContext.cpp
/// Get a dialect for the provided namespace and TypeID: abort the program if a
/// dialect exist for this namespace with different TypeID. Returns a pointer to
/// the dialect owned by the context.
Dialect *
MLIRContext::getOrLoadDialect(StringRef dialectNamespace, TypeID dialectID,
                              function_ref<std::unique_ptr<Dialect>()> ctor) {
  auto &impl = getImpl();

  // Call BuiltinDialect constructor
  impl.loadedDialects[dialectNamespace] = ctor();
  Dialect *dialect = dialectOwned.get();
  assert(dialect && "dialect ctor failed");

  // Apply any extensions to this newly loaded dialect.
  impl.dialectsRegistry.applyExtensions(dialect);
  return dialect;

  return dialect.get();
}

// BuiltinDialect constructor is defined in
// llvm-project/build/tools/mlir/include/mlir/IR/BuiltinDialect.cpp.inc
namespace mlir {

BuiltinDialect::BuiltinDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<BuiltinDialect>())
    
     {
  
  initialize();
}

BuiltinDialect::~BuiltinDialect() = default;

} // namespace mlir


// llvm-project/mlir/lib/IR/BuiltinDialect.cpp
void BuiltinDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerLocationAttributes();
  addOperations<
#define GET_OP_LIST
#include "mlir/IR/BuiltinOps.cpp.inc"
      >();

  auto &blobInterface = addInterface<BuiltinBlobManagerInterface>();
  addInterface<BuiltinOpAsmDialectInterface>(blobInterface);
  builtin_dialect_detail::addBytecodeInterface(this);
}

void BuiltinDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/IR/BuiltinTypes.cpp.inc"
      >();
}
// llvm-project/build/tools/mlir/include/mlir/IR/BuiltinTypes.cpp.inc
// includes `::mlir::IntegerType`

// mlir::Dialect::addTypes<...>()
// llvm-project/mlir/include/mlir/IR/Dialect.h
  /// Register a set of type classes with this dialect.
  template <typename... Args>
  void addTypes() {
    // This initializer_list argument pack expansion is essentially equal to
    // using a fold expression with a comma operator. Clang however, refuses
    // to compile a fold expression with a depth of more than 256 by default.
    // There seem to be no such limitations for initializer_list.
    (void)std::initializer_list<int>{0, (addType<Args>(), 0)...};
  }

  /// Register a type instance with this dialect.
  template <typename T>
  void addType() {
    // Add this type to the dialect and register it with the uniquer.
    addType(T::getTypeID(), AbstractType::get<T>(*this));
    detail::TypeUniquer::registerType<T>(context);
  }

// Say T = IntegerType, then AbstractType::get<T>()
// llvm-project/mlir/include/mlir/IR/TypeSupport.h
class AbstractType {
public:
  /// This method is used by Dialect objects when they register the list of
  /// types they contain.
  template <typename T>
  static AbstractType get(Dialect &dialect) {
    return AbstractType(dialect, T::getInterfaceMap(), T::getHasTraitFn(),
                        T::getWalkImmediateSubElementsFn(),
                        T::getReplaceImmediateSubElementsFn(), T::getTypeID(),
                        T::name);
  }

private:
  AbstractType(Dialect &dialect, detail::InterfaceMap &&interfaceMap,
               HasTraitFn &&hasTrait,
               WalkImmediateSubElementsFn walkImmediateSubElementsFn,
               ReplaceImmediateSubElementsFn replaceImmediateSubElementsFn,
               TypeID typeID, StringRef name)
      : dialect(dialect), interfaceMap(std::move(interfaceMap)),
        hasTraitFn(std::move(hasTrait)),
        walkImmediateSubElementsFn(walkImmediateSubElementsFn),
        replaceImmediateSubElementsFn(replaceImmediateSubElementsFn),
        typeID(typeID), name(name) {}
}

// llvm-project/mlir/lib/IR/MLIRContext.cpp
void Dialect::addType(TypeID typeID, AbstractType &&typeInfo) {
  auto &impl = context->getImpl();
  assert(impl.multiThreadedExecutionContext == 0 &&
         "Registering a new type kind while in a multi-threaded execution "
         "context");
  auto *newInfo =
      new (impl.abstractDialectSymbolAllocator.Allocate<AbstractType>())
          AbstractType(std::move(typeInfo));
  if (!impl.registeredTypes.insert({typeID, newInfo}).second)
    llvm::report_fatal_error("Dialect Type already registered.");
  if (!impl.nameToType.insert({newInfo->getName(), newInfo}).second)
    llvm::report_fatal_error("Dialect Type with name " + newInfo->getName() +
                             " is already registered.");
}
// NOTE1: An AbstractType object for IntegerType is regstered to MLIRContext object

// detail::TypeUniquer::registerType<T>(context)
// T = IntegerType
// llvm-project/mlir/include/mlir/IR/TypeSupport.h
  /// Register a type instance T with the uniquer.
  template <typename T>
  static void registerType(MLIRContext *ctx) {
    registerType<T>(ctx, T::getTypeID());
  }

  /// Register a parametric type instance T with the uniquer.
  /// The use of this method is in general discouraged in favor of
  /// 'registerType<T>(ctx)'.
  template <typename T>
  static std::enable_if_t<
      !std::is_same<typename T::ImplType, TypeStorage>::value>
  registerType(MLIRContext *ctx, TypeID typeID) {
    ctx->getTypeUniquer().registerParametricStorageType<typename T::ImplType>(
        typeID);
  }

// StorageUniquer::registerParametricStorageType(mlir::TypeID id)
// llvm-project/mlir/include/mlir/Support/StorageUniquer.h
class StorageUniquer {
public:
  /// Register a new parametric storage class, this is necessary to create
  /// instances of this class type. `id` is the type identifier that will be
  /// used to identify this type when creating instances of it via 'get'.
  template <typename Storage>
  void registerParametricStorageType(TypeID id) {
    // If the storage is trivially destructible, we don't need a destructor
    // function.
    if constexpr (std::is_trivially_destructible_v<Storage>)
      return registerParametricStorageTypeImpl(id, nullptr);
    registerParametricStorageTypeImpl(id, [](BaseStorage *storage) {
      static_cast<Storage *>(storage)->~Storage();
    });
  }
}

// llvm-project/mlir/lib/Support/StorageUniquer.cpp
struct StorageUniquerImpl {
  using BaseStorage = StorageUniquer::BaseStorage;
  using StorageAllocator = StorageUniquer::StorageAllocator;

  /// Main allocator used for uniquing singleton instances, and other state when
  /// thread safety is guaranteed.
  StorageAllocator allocator;

  /// Map of type ids to the storage uniquer to use for registered objects.
  DenseMap<TypeID, std::unique_ptr<ParametricStorageUniquer>> parametricUniquers;
}

/// Implementation for registering an instance of a derived type with
/// parametric storage.
void StorageUniquer::registerParametricStorageTypeImpl(
    TypeID id, function_ref<void(BaseStorage *)> destructorFn) {
  impl->parametricUniquers.try_emplace(
      id, std::make_unique<ParametricStorageUniquer>(destructorFn));
}
// NOTE2: Create a kv-pair: {id, std::unique_ptr<ParametricStorageUniquer>}
```
