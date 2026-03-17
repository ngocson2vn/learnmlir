#ifndef TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonGPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

// LinearLayoutCache Utils
using CacheKey = std::tuple<std::vector<int64_t>, mlir::Attribute>;

namespace llvm {
template <typename T> size_t hash_value(const std::vector<T> &vec) {
  return hash_combine_range(vec.begin(), vec.end());
}
} // namespace llvm

namespace std {
template <> struct hash<CacheKey> {
  size_t operator()(const CacheKey &key) const noexcept {
    using llvm::hash_value;
    size_t seed = 0;
    std::apply(
        [&seed](const auto &...elems) {
          ((seed = llvm::hash_combine(seed, hash_value(elems))), ...);
        },
        key);
    return seed;
  }
};
} // namespace std


namespace mlir::triton::gpu {

constexpr static char AttrMaxRegistersName[] = "ttg.maxnreg";
constexpr static char AttrNumWarpsName[] = "ttg.num-warps";
constexpr static char AttrNumCTAsName[] = "ttg.num-ctas";
constexpr static char AttrTargetName[] = "ttg.target";
constexpr static char AttrNumThreadsPerWarp[] = "ttg.threads-per-warp";
// FIXME: rename to match above
constexpr static char kPartitionAttrName[] = "ttg.partition";
constexpr static char kPartitionOutputsAttrName[] = "ttg.partition.outputs";
constexpr static char kPartitionStagesAttrName[] = "ttg.partition.stages";
constexpr static char kWarpSpecializeTagAttrName[] = "ttg.warp_specialize.tag";

template <typename Key, typename Value> class Cache {
public:
  std::optional<Value> get(const Key &key) {
    std::shared_lock lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  void set(Key key, Value result) {
    std::scoped_lock lock(mutex);
    cache.emplace(std::move(key), std::move(result));
  }

private:
  std::unordered_map<Key, Value> cache;
  llvm::sys::SmartRWMutex<true> mutex;
};

using LinearLayoutCache = Cache<CacheKey, LinearLayout>;
using LinearEncodingCache = Cache<CacheKey, LinearEncodingAttr>;

} // namespace mlir::triton::gpu

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Dialect.h.inc"

namespace mlir::triton::gpu {

// Returns the "logical" shape per CTA
SmallVector<int64_t> getShapePerCTA(ArrayRef<unsigned> CTASplitNum,
                                    ArrayRef<int64_t> shape);
SmallVector<int64_t> getShapePerCTA(Attribute layout, ArrayRef<int64_t> shape);
SmallVector<int64_t> getShapePerCTA(Type type);

// Return the order that represents that the batch is in row-major or
// column-major order for a batch of matrices of shape [*, m, n] with
// len(shape) == rank.
SmallVector<unsigned> getMatrixOrder(unsigned rank, bool rowMajor);

}

#endif // TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_