<!-- TOC START -->
- [MLIR Uniquing Process](#mlir-uniquing-process)
    - [1. Packing the Parameters (`KeyTy`)](#1-packing-the-parameters-keyty)
    - [2. Hashing the Key](#2-hashing-the-key)
    - [3. The Thread-Safe Table Lookup (Read Phase)](#3-the-thread-safe-table-lookup-read-phase)
    - [4. The Equality Check (Handling Collisions)](#4-the-equality-check-handling-collisions)
    - [5. The Allocation Phase (Write Phase)](#5-the-allocation-phase-write-phase)
    - [6. Memory Allocation via BumpPtrAllocator](#6-memory-allocation-via-bumpptrallocator)
    - [7. Construction and Hash Table Insertion](#7-construction-and-hash-table-insertion)
    - [8. Wrapping and Returning](#8-wrapping-and-returning)
- [IntegerType](#integertype)
  - [What is `Base` class?](#what-is-base-class)
  - [Pre-register IntegerType types](#pre-register-integertype-types)
    - [Step 1: Register IntegerType into MLIRContext](#step-1-register-integertype-into-mlircontext)
    - [Step 2: Pre-create an instance of IntegerType](#step-2-pre-create-an-instance-of-integertype)
  - [IntegerType::get](#integertypeget)
<!-- TOC END -->


# MLIR Uniquing Process
Here is the step-by-step lifecycle of what happens under the hood when you execute `IntegerType::get(context, 32)`. 

Behind the scenes, MLIR delegates this request to a highly optimized, thread-safe component inside the `MLIRContext` called the **`StorageUniquer`**.



Here is the exact sequence of events:

### 1. Packing the Parameters (`KeyTy`)
First, the `IntegerType::get` method packs your requested parameters into a standard format. In MLIR C++ API terms, this is called the `KeyTy` (Key Type). For a standard signless integer, the `KeyTy` is essentially just an `unsigned int` representing the bitwidth (32).

### 2. Hashing the Key
The `StorageUniquer` takes this `KeyTy` and runs it through an LLVM hashing function (like `llvm::hash_value`). This produces a numerical hash code. The hash code is used to quickly identify which bucket in the `MLIRContext`'s internal hash table this type *should* live in.

### 3. The Thread-Safe Table Lookup (Read Phase)
Because MLIR is designed to compile code concurrently across many threads, the `MLIRContext` must be thread-safe. 
* The `StorageUniquer` acquires a **read lock**.
* It jumps to the specific bucket in the hash table using the calculated hash.
* It searches through the bucket to see if an object with this exact hash already exists.

### 4. The Equality Check (Handling Collisions)
Hashes can collide (two different sets of parameters could theoretically produce the same hash). If the `StorageUniquer` finds an existing `Storage` object with a matching hash, it must verify it's truly the right type. 
It calls an `operator==` method on the existing storage object, comparing its internal data to your `KeyTy` (e.g., *Is the existing integer storage actually 32 bits?*).

* **Cache Hit:** If the hash matches *and* the equality check passes, the `StorageUniquer` unlocks and immediately returns a pointer to this existing object. The process ends here.
* **Cache Miss:** If no object is found, the process must proceed to create one.

### 5. The Allocation Phase (Write Phase)
If this is the very first time an `i32` has been requested in this context, the `StorageUniquer` upgrades to a **write lock** to ensure no other threads interfere while it allocates memory.

Because another thread might have squeezed in and created the `i32` type in the microsecond between the read lock and the write lock, MLIR performs a "double-checked lock." It quickly checks the hash table one more time. Assuming it's still not there, it proceeds to allocate.

### 6. Memory Allocation via BumpPtrAllocator
Standard C++ `new` or `malloc` is too slow for compiler infrastructure because it carries significant overhead. Instead, the `MLIRContext` uses an `llvm::BumpPtrAllocator`. 
* This allocator grabs huge chunks (slabs) of memory from the operating system at once.
* When MLIR needs memory for the `IntegerTypeStorage`, the allocator simply advances a pointer forward by the required number of bytes and returns that address. This is nearly instantaneous.

### 7. Construction and Hash Table Insertion
Now that it has raw memory, MLIR calls the `construct` method for the `IntegerTypeStorage` class. It initializes the memory, setting its internal bitwidth field to 32. Finally, the `StorageUniquer` inserts the pointer to this newly minted object into the hash table so future calls can find it. The write lock is released.

### 8. Wrapping and Returning
The pointer to the `IntegerTypeStorage` object is passed back to `IntegerType::get`. The C++ API wraps this raw pointer in the lightweight, value-typed `IntegerType` C++ class and hands it back to your code.
<br/>
<br/>

Now, pick `IntegerType` as an example:
# IntegerType
llvm-project/build/tools/mlir/include/mlir/IR/BuiltinTypes.h.inc
```C++
class IntegerType : public ::mlir::Type::TypeBase<IntegerType, ::mlir::Type, detail::IntegerTypeStorage, ::mlir::VectorElementTypeInterface::Trait> {
public:
  using Base::Base;
```

## What is `Base` class?
llvm-project/mlir/include/mlir/IR/Types.h
```C++
class Type {
public:
  /// Utility class for implementing types.
  template <typename ConcreteType, typename BaseType, typename StorageType,
            template <typename T> class... Traits>
  using TypeBase = detail::StorageUserBase<ConcreteType, BaseType, StorageType,
                                           detail::TypeUniquer, Traits...>;


// llvm-project/mlir/include/mlir/IR/StorageUniquerSupport.h
template <typename ConcreteT, typename BaseT, typename StorageT,
          typename UniquerT, template <typename T> class... Traits>
class StorageUserBase : public BaseT, public Traits<ConcreteT>... {
public:
  using BaseT::BaseT;

  /// Utility declarations for the concrete attribute class.
  using Base = StorageUserBase<ConcreteT, BaseT, StorageT, UniquerT, Traits...>;

// === INHERITANCE HIERARCHY ===
// IntegerType 
//   -> detail::StorageUserBase<IntegerType, Type, detail::IntegerTypeStorage, detail::TypeUniquer, VectorElementTypeInterface::Trait>
//     -> Type
```

## Pre-register IntegerType types
### Step 1: Register an AbstractType object for IntegerType to MLIRContext object
See [mlir_builtin_types.md](./mlir_builtin_types.md)

### Step 1: Register a ParametricStorageUniquer object for IntegerType to MLIRContextImpl::typeUniquer
See [mlir_builtin_types.md](./mlir_builtin_types.md)

Summary:
```
MLIRContextImpl
  -> typeUniquer (is of StorageUniquer)
    -> {IntegerType::getTypeID(), std::unique_ptr<ParametricStorageUniquer>}
```


### Step 3: Pre-create an instance of IntegerType

`IntegerType` has an underlying `IntegerTypeStorage` object. Why?
MLIRContext
  -> TypeUniquer
    -> StorageUniquer
       - params -> getKey -> getHash -> hashValue
       - create ctorFn
      -> ParametricStorageUniquer
         - hashValue -> Shard -> DenseMap
         - call ctorFn() to create a IntegerTypeStorage object in StorageUniquerImpl::StorageAllocator
         

```C++
// llvm-project/mlir/lib/IR/MLIRContext.cpp
MLIRContext::MLIRContext(const DialectRegistry &registry, Threading setting)
    : impl(new MLIRContextImpl(setting == Threading::ENABLED &&
                               !isThreadingGloballyDisabled())) {
  impl->int32Ty =
      TypeUniquer::get<IntegerType>(this, 32, IntegerType::Signless);
}

// llvm-project/mlir/include/mlir/IR/TypeSupport.h
struct TypeUniquer {
  /// Get an uniqued instance of a type T.
  template <typename T, typename... Args>
  static T get(MLIRContext *ctx, Args &&...args) {
    return getWithTypeID<T, Args...>(ctx, T::getTypeID(),
                                     std::forward<Args>(args)...);
  }

  /// Get an uniqued instance of a parametric type T.
  /// The use of this method is in general discouraged in favor of
  /// 'get<T, Args>(ctx, args)'.
  template <typename T, typename... Args>
  static std::enable_if_t<
      !std::is_same<typename T::ImplType, TypeStorage>::value, T>
  getWithTypeID(MLIRContext *ctx, TypeID typeID, Args &&...args) {
#ifndef NDEBUG
    if (!ctx->getTypeUniquer().isParametricStorageInitialized(typeID))
      llvm::report_fatal_error(
          llvm::Twine("can't create type '") + llvm::getTypeName<T>() +
          "' because storage uniquer isn't initialized: the dialect was likely "
          "not loaded, or the type wasn't added with addTypes<...>() "
          "in the Dialect::initialize() method.");
#endif
    return ctx->getTypeUniquer().get<typename T::ImplType>(
        [&, typeID](TypeStorage *storage) {
          storage->initialize(AbstractType::lookup(typeID, ctx));
        },
        typeID, std::forward<Args>(args)...);
  }

// Since T = IntegerType which inherits from Type::TypeBase<IntegerType, ::mlir::Type, detail::IntegerTypeStorage, ::mlir::VectorElementTypeInterface::Trait>

// Substituting arguments into the alias template Type::TypeBase, we obtain
class Type {
public:
  using TypeBase = detail::StorageUserBase<IntegerType, Type, detail::IntegerTypeStorage, detail::TypeUniquer, VectorElementTypeInterface::Trait>;
}

// So actually IntegerType inherits from detail::StorageUserBase<IntegerType, Type, detail::IntegerTypeStorage, detail::TypeUniquer, VectorElementTypeInterface::Trait>

// Substituting arguments into the class template detail::StorageUserBase, we obtain
class StorageUserBase : public Type, public VectorElementTypeInterface::Trait {
public:
  using BaseT::BaseT;

  /// Utility declarations for the concrete attribute class.
  using Base = StorageUserBase<IntegerType, Type, detail::IntegerTypeStorage, detail::TypeUniquer, VectorElementTypeInterface::Trait>;
  using ImplType = detail::IntegerTypeStorage;
  using HasTraitFn = bool (*)(TypeID);

  /// Return a unique identifier for the concrete type.
  static TypeID getTypeID() { return TypeID::get<IntegerType>(); }

// So T::ImplType = IntegerType::ImplType = detail::IntegerTypeStorage

// TypeUniquer::get<IntegerType>(this, 32, IntegerType::Signless) becomes
struct TypeUniquer {
  /// Get an uniqued instance of a type T.
  static IntegerType get(MLIRContext *ctx, int&& arg0, SignednessSemantics&& arg1) {
    return getWithTypeID<IntegerType, int, SignednessSemantics>(ctx, T::getTypeID(), std::forward<int>(arg0), std::forward<SignednessSemantics>(arg1));
  }

  IntegerType getWithTypeID(MLIRContext *ctx, TypeID typeID, int&& arg0, SignednessSemantics&& arg1) {
#ifndef NDEBUG
    if (!ctx->getTypeUniquer().isParametricStorageInitialized(typeID))
      llvm::report_fatal_error(
          llvm::Twine("can't create type '") + llvm::getTypeName<T>() +
          "' because storage uniquer isn't initialized: the dialect was likely "
          "not loaded, or the type wasn't added with addTypes<...>() "
          "in the Dialect::initialize() method.");
#endif
    return ctx->getTypeUniquer().get<detail::IntegerTypeStorage>(
        [&, typeID](TypeStorage *storage) {
          storage->initialize(AbstractType::lookup(typeID, ctx));
        },
        typeID, std::forward<int>(arg0), std::forward<SignednessSemantics>(arg1));
  }

// T::getTypeID() = IntegerType::getTypeID()

// ctx->getTypeUniquer()
StorageUniquer &MLIRContext::getTypeUniquer() { return getImpl().typeUniquer; }

// typeUniquer is a data member of MLIRContextImpl, which is of type of `StorageUniquer`.
// StorageUniquer has a StorageAllocator allocator object which is responsible for allocating memory for hosting TypeStorage objects.
class MLIRContextImpl {
public:
  StorageUniquer typeUniquer;
}

// StorageUniquer::get<T>()
// llvm-project/mlir/include/mlir/Support/StorageUniquer.h
class StorageUniquer {
public:
  /// Gets a uniqued instance of 'Storage'. 'id' is the type id used when
  /// registering the storage instance. 'initFn' is an optional parameter that
  /// can be used to initialize a newly inserted storage instance. This function
  /// is used for derived types that have complex storage or uniquing
  /// constraints.
  template <typename Storage, typename... Args>
  Storage *get(function_ref<void(Storage *)> initFn, TypeID id,
               Args &&...args) {
    // Construct a value of the derived key type.
    auto derivedKey = getKey<Storage>(std::forward<Args>(args)...);

    // Create a hash of the derived key.
    unsigned hashValue = getHash<Storage>(derivedKey);

    // Generate an equality function for the derived storage.
    auto isEqual = [&derivedKey](const BaseStorage *existing) {
      return static_cast<const Storage &>(*existing) == derivedKey;
    };

    // Generate a constructor function for the derived storage.
    auto ctorFn = [&](StorageAllocator &allocator) {
      auto *storage = Storage::construct(allocator, std::move(derivedKey));
      if (initFn)
        initFn(storage);
      return storage;
    };

    // Get an instance for the derived storage.
    return static_cast<Storage *>(
        getParametricStorageTypeImpl(id, hashValue, isEqual, ctorFn));
  }

private:
  /// The internal implementation class.
  std::unique_ptr<detail::StorageUniquerImpl> impl;
}

// Storage = detail::IntegerTypeStorage

// StorageUniquer::getParametricStorageTypeImpl
// llvm-project/mlir/lib/Support/StorageUniquer.cpp
StorageUniquer::StorageUniquer() : impl(new StorageUniquerImpl()) {}

/// Implementation for getting/creating an instance of a derived type with
/// parametric storage.
auto StorageUniquer::getParametricStorageTypeImpl(
    TypeID id, unsigned hashValue,
    function_ref<bool(const BaseStorage *)> isEqual,
    function_ref<BaseStorage *(StorageAllocator &)> ctorFn) -> BaseStorage * {
  return impl->getOrCreate(id, hashValue, isEqual, ctorFn);
}
// id is an instance of TypeID for IntegerType


struct StorageUniquerImpl {
  using BaseStorage = StorageUniquer::BaseStorage;
  using StorageAllocator = StorageUniquer::StorageAllocator;

  /// Get or create an instance of a parametric type.
  BaseStorage *
  getOrCreate(TypeID id, unsigned hashValue,
              function_ref<bool(const BaseStorage *)> isEqual,
              function_ref<BaseStorage *(StorageAllocator &)> ctorFn) {
    assert(parametricUniquers.count(id) &&
           "creating unregistered storage instance");
    ParametricStorageUniquer &storageUniquer = *parametricUniquers[id];
    return storageUniquer.getOrCreate(
        threadingIsEnabled, hashValue, isEqual,
        [&] { return ctorFn(getThreadSafeAllocator()); });
  }

}


/// This class represents a uniquer for storage instances of a specific type
/// that has parametric storage. It contains all of the necessary data to unique
/// storage instances in a thread safe way. This allows for the main uniquer to
/// bucket each of the individual sub-types removing the need to lock the main
/// uniquer itself.
class ParametricStorageUniquer {
public:
  using BaseStorage = StorageUniquer::BaseStorage;
  using StorageAllocator = StorageUniquer::StorageAllocator;

  /// A lookup key for derived instances of storage objects.
  struct LookupKey {
    /// The known hash value of the key.
    unsigned hashValue;

    /// An equality function for comparing with an existing storage instance.
    function_ref<bool(const BaseStorage *)> isEqual;
  };

private:
  /// Get or create an instance of a param derived type in an thread-unsafe
  /// fashion.
  BaseStorage *getOrCreateUnsafe(Shard &shard, LookupKey &key,
                                 function_ref<BaseStorage *()> ctorFn) {
    auto existing = shard.instances.insert_as({key.hashValue}, key);
    BaseStorage *&storage = existing.first->storage;
    if (existing.second)
      storage = ctorFn();
    return storage;
  }

public:
  /// Get or create an instance of a parametric type.
  BaseStorage *getOrCreate(bool threadingIsEnabled, unsigned hashValue,
                           function_ref<bool(const BaseStorage *)> isEqual,
                           function_ref<BaseStorage *()> ctorFn) {
    Shard &shard = getShard(hashValue);
    ParametricStorageUniquer::LookupKey lookupKey{hashValue, isEqual};
    if (!threadingIsEnabled)
      return getOrCreateUnsafe(shard, lookupKey, ctorFn);

    // Check for a instance of this object in the local cache.
    auto localIt = localCache->insert_as({hashValue}, lookupKey);
    BaseStorage *&localInst = localIt.first->storage;
    if (localInst)
      return localInst;

    // Check for an existing instance in read-only mode.
    {
      llvm::sys::SmartScopedReader<true> typeLock(shard.mutex);
      auto it = shard.instances.find_as(lookupKey);
      if (it != shard.instances.end())
        return localInst = it->storage;
    }

    // Acquire a writer-lock so that we can safely create the new storage
    // instance.
    llvm::sys::SmartScopedWriter<true> typeLock(shard.mutex);
    return localInst = getOrCreateUnsafe(shard, lookupKey, ctorFn);
  }

private:
  /// Return the shard used for the given hash value.
  Shard &getShard(unsigned hashValue) {
    // Get a shard number from the provided hashvalue.
    unsigned shardNum = hashValue & (numShards - 1);

    // Try to acquire an already initialized shard.
    Shard *shard = shards[shardNum].load(std::memory_order_acquire);
    if (shard)
      return *shard;

    // Otherwise, try to allocate a new shard.
    Shard *newShard = new Shard();
    if (shards[shardNum].compare_exchange_strong(shard, newShard))
      return *newShard;

    // If one was allocated before we can initialize ours, delete ours.
    delete newShard;
    return *shard;
  }

  /// A thread local cache for storage objects. This helps to reduce the lock
  /// contention when an object already existing in the cache.
  ThreadLocalCache<StorageTypeSet> localCache;

  /// A set of uniquer shards to allow for further bucketing accesses for
  /// instances of this storage type. Each shard is lazily initialized to reduce
  /// the overhead when only a small amount of shards are in use.
  std::unique_ptr<std::atomic<Shard *>[]> shards;

  /// The number of available shards.
  size_t numShards;
}

// Eventually, TypeUniquer::getWithTypeID<IntegerType, Args...>(MLIRContext *ctx, TypeID typeID, Args &&...args) returns an IntegerTypeStorage* pointer.
// IntegerType inheritance hierarchy:
//   -> detail::StorageUserBase<IntegerType, Type, detail::IntegerTypeStorage, detail::TypeUniquer, VectorElementTypeInterface::Trait>
//     -> Type
// Since IntegerType has `using Base::Base` and detail::StorageUserBase has `using BaseT::BaseT`, IntegerType inherits the parameterized constructor `Type(const ImplType *impl)` of the Type class.
// The compiler will call `Type(const ImplType *impl)` directly to construct a Type subobject and then initialize data members of detail::StorageUserBase and IntegerType, respectively.
// The compiler skips calling default constructors of detail::StorageUserBase and IntegerType.
// Finally an IntegerType object is constructed.
```


## IntegerType::get
When the client code `auto i32Type = mlir::IntegerType::get(&context, 32);` is executed, the following process will happen:<br/><br/>

llvm-project/mlir/lib/IR/MLIRContext.cpp
```C++
IntegerType IntegerType::get(MLIRContext *context, unsigned width,
                             IntegerType::SignednessSemantics signedness) {
  if (auto cached = getCachedIntegerType(width, signedness, context))
    return cached;
  return Base::get(context, width, signedness);
}

/// Return an existing integer type instance if one is cached within the
/// context.
static IntegerType
getCachedIntegerType(unsigned width,
                     IntegerType::SignednessSemantics signedness,
                     MLIRContext *context) {
  if (signedness != IntegerType::Signless)
    return IntegerType();

  switch (width) {
  case 1:
    return context->getImpl().int1Ty;
  case 8:
    return context->getImpl().int8Ty;
  case 16:
    return context->getImpl().int16Ty;
  case 32:
    return context->getImpl().int32Ty;
  case 64:
    return context->getImpl().int64Ty;
  case 128:
    return context->getImpl().int128Ty;
  default:
    return IntegerType();
  }
}
```
