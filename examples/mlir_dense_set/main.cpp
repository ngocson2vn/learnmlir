#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Debug.h"

using BaseStorage = mlir::StorageUniquer::BaseStorage;

struct HashedStorage {
  HashedStorage(unsigned hashValue = 0, BaseStorage *storage = nullptr)
      : hashValue(hashValue), storage(storage) {}
  unsigned hashValue;
  BaseStorage *storage;
};

struct LookupKey {
  /// The known hash value of the key.
  unsigned hashValue;

  /// An equality function for comparing with an existing storage instance.
  mlir::function_ref<bool(const BaseStorage *)> isEqual;
};

struct StorageKeyInfo {
  static inline HashedStorage getEmptyKey() {
    return HashedStorage(0, mlir::DenseMapInfo<BaseStorage *>::getEmptyKey());
  }
  static inline HashedStorage getTombstoneKey() {
    return HashedStorage(0, mlir::DenseMapInfo<BaseStorage *>::getTombstoneKey());
  }

  static inline unsigned getHashValue(const HashedStorage &key) {
    return key.hashValue;
  }
  static inline unsigned getHashValue(const LookupKey &key) {
    return key.hashValue;
  }

  static inline bool isEqual(const HashedStorage &lhs,
                              const HashedStorage &rhs) {
    return lhs.storage == rhs.storage;
  }
  static inline bool isEqual(const LookupKey &lhs, const HashedStorage &rhs) {
    if (isEqual(rhs, getEmptyKey()) || isEqual(rhs, getTombstoneKey()))
      return false;
    // Invoke the equality function on the lookup key.
    return lhs.isEqual(rhs.storage);
  }
};

struct IntegerTypeStorage : public BaseStorage {
  IntegerTypeStorage(unsigned width,
                     mlir::IntegerType::SignednessSemantics signedness)
      : width(width), signedness(signedness) {}

  /// The hash key used for uniquing.
  using KeyTy = std::tuple<unsigned, mlir::IntegerType::SignednessSemantics>;

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  bool operator==(const KeyTy &key) const {
    return KeyTy(width, signedness) == key;
  }

  // static IntegerTypeStorage *construct(TypeStorageAllocator &allocator,
  //                                      KeyTy key) {
  //   return new (allocator.allocate<IntegerTypeStorage>())
  //       IntegerTypeStorage(std::get<0>(key), std::get<1>(key));
  // }

  // KeyTy getAsKey() const { return KeyTy(width, signedness); }

  unsigned width : 30;
  mlir::IntegerType::SignednessSemantics signedness : 2;
};

template<typename T>
void printType() {
  std::string type = __PRETTY_FUNCTION__;
  llvm::outs() << type << "\n";
}

int main(int argc, char **argv) {
  // Set up the MLIR context
  mlir::MLIRContext context;

  mlir::DenseSet<HashedStorage, StorageKeyInfo> storages;
  unsigned hashValue = 1000;
  auto derivedKey = IntegerTypeStorage::KeyTy(32, mlir::IntegerType::Signless);
  auto isEqual = [&derivedKey](const BaseStorage *existing) {
    return static_cast<const IntegerTypeStorage &>(*existing) == derivedKey;
  };

  LookupKey lookupKey{hashValue, isEqual};
  HashedStorage s(hashValue, new IntegerTypeStorage(32, mlir::IntegerType::Signless));
  auto ret1 = storages.insert_as(s, lookupKey);
  auto& first = *ret1.first;
  printType<decltype(first)>();
  // void printType() [T = HashedStorage &]
  // So `first` is a HashedStorage&

  llvm::outs() << "ret1.first = " << (*ret1.first).storage << "\n";

  // If the insertion is successful, then `.second` should be true
  llvm::outs() << "ret1.second = " << (ret1.second ? "true" : "false") << "\n\n";

  auto ret2 = storages.insert_as(s, lookupKey);
  llvm::outs() << "ret2.first = " << (*ret2.first).storage << "\n";

  // Since the `s` element exists in the set, the insertion should fail.
  // Therefore `.second` should be false
  llvm::outs() << "ret2.second = " << (ret2.second ? "true" : "false") << "\n";

  return 0;
}
