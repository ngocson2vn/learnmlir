#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LayoutUtils.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/StrUtil.h"

// Include TableGen'erated code
#include "triton/Dialect/TritonGPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

static SmallVector<unsigned>
basesPerDimImpl(const LinearLayout::BasesT &namedBases, StringAttr dimName,
                size_t rank, bool skipBroadcast = true);

// Utility
namespace mlir {
namespace triton {
namespace gpu {

LinearEncodingAttr TritonGPUDialect::toLinearEncoding(ArrayRef<int64_t> shape,
                                                      Attribute layout) {
  // LinearEncoding is a DistributedLayout
  std::vector<int64_t> allocationShape;
  CacheKey key{std::vector<int64_t>(shape.begin(), shape.end()), layout};
  if (auto result = leCache.get(key)) {
    return *result;
  }
  auto linearLayout = toLinearLayout(shape, layout);
  auto linearEncoding =
      LinearEncodingAttr::get(layout.getContext(), std::move(linearLayout));
  leCache.set(key, linearEncoding);
  return linearEncoding;
}

LinearEncodingAttr toLinearEncoding(DistributedEncodingTrait layout,
                                    ArrayRef<int64_t> shape) {
  auto *ctx = layout.getContext();
  return ctx->getLoadedDialect<TritonGPUDialect>()->toLinearEncoding(shape,
                                                                     layout);
}

LinearEncodingAttr toLinearEncoding(RankedTensorType type) {
  auto *ctx = type.getContext();
  return ctx->getLoadedDialect<TritonGPUDialect>()->toLinearEncoding(
      type.getShape(), type.getEncoding());
}

/* Utility function used by get.*Order methods of SliceEncodingAttr.
 * Erase dim and decrease all values larger than dim by 1.
 * Example:    order = [0, 2, 4, 3, 1], dim = 2
 *          resOrder = [0,    3, 2, 1]
 */
static SmallVector<unsigned> eraseOrder(ArrayRef<unsigned> order,
                                        unsigned dim) {
  unsigned rank = order.size();
  assert(dim < rank && "Invalid dim to erase");
  SmallVector<unsigned> resOrder;
  for (unsigned i : order)
    if (i < dim)
      resOrder.push_back(i);
    else if (i > dim)
      resOrder.push_back(i - 1);
  return resOrder;
}

SmallVector<unsigned> getMatrixOrder(unsigned rank, bool rowMajor) {
  // Return the order that represents that the batch is in row-major or
  // column-major order for a batch of matrices of shape [*, m, n] with
  // len(shape) == rank.
  SmallVector<unsigned> order(rank);
  if (rank < 2) {
    return order;
  }
  std::iota(order.rbegin(), order.rend(), 0);
  if (!rowMajor) {
    std::swap(order[0], order[1]);
  }
  return order;
}

SmallVector<unsigned> getOrderForDotOperand(unsigned opIdx, unsigned rank,
                                            bool kContig) {
  // kContig: if true, the matrix is fastest-running on k,
  //         otherwise it is on m (resp. n)
  // opIdx=0: [*batch, m, k]
  // opIdx=1: [*batch, k, n]
  assert(opIdx == 0 || opIdx == 1);
  auto rowMajor = bool(opIdx) != kContig;
  return getMatrixOrder(rank, rowMajor);
}

static LogicalResult
verifyLayoutOrder(function_ref<InFlightDiagnostic()> emitError,
                  ArrayRef<unsigned> order) {
  if (!isPermutationOfIota(order)) {
    return emitError()
           << "order must be a permutation of 0..(rank-1), but was [" << order
           << "]";
  }
  return success();
}

SmallVector<unsigned> orderPerDimImpl(const LinearLayout &ll,
                                      StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) {
  assert(ll.getBases().contains(dimName));
  const auto &bases = ll.getBases().find(dimName)->second;
  llvm::SetVector<unsigned> order;
  auto nonZero = [](auto val) { return val != 0; };
  for (const auto &basis : bases) {
    // Bases can have one or zero non-zero elements
    // Skip a basis if it's broadcasting (all zeros)
    // e.g. warps for DotOperandEncodingAttr (see ampereDotToLinearLayout)
    auto it = std::find_if(basis.begin(), basis.end(), nonZero);
    if (it != basis.end()) {
      auto i = it - basis.begin();
      order.insert(i);
    }
  }
  // If any dim is missing, we add them in the defaultOrder
  for (auto i : defaultOrder) {
    order.insert(i);
  }
  return order.takeVector();
}

SmallVector<unsigned>
LinearEncodingAttr::basesPerDim(StringAttr dimName, bool skipBroadcast) const {
  auto ll = getLinearLayout();
  auto rank = ll.getNumOutDims();
  return basesPerDimImpl(ll.getBases(), dimName, rank, skipBroadcast);
}

LogicalResult
CTAEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        LinearLayout linearLayout) {
  if (linearLayout.getNumInDims() != 1) {
    return emitError() << "CTA encoding must have exactly one input dimension "
                          "named 'block'.";
  }
  auto dim = *linearLayout.getInDimNames().begin();
  auto ctx = dim.getContext();
  if (dim != StringAttr::get(ctx, "block")) {
    return emitError() << "CTA encoding must have exactly one input dimension "
                          "named 'block'.";
  }

  auto outDimNames = linearLayout.getOutDimNames();
  auto expected = standardOutDimNames(ctx, linearLayout.getNumOutDims());
  if (!llvm::equal(outDimNames, expected)) {
    return emitError() << "CTA encoding output dims must be [dim0, dim1, ...], "
                          "but got ["
                       << outDimNames << "].";
  }

  return success();
}

CTAEncodingAttr CTAEncodingAttr::getDefault(MLIRContext *ctx, int rank) {
  auto kBlock = StringAttr::get(ctx, "block");
  LinearLayout::BasesT bases;
  bases[kBlock] = {};
  auto dims = standardOutDimNames(ctx, rank);
  return get(ctx, LinearLayout(bases, dims));
}

CTAEncodingAttr CTAEncodingAttr::fromSplitParams(MLIRContext *ctx,
                                                 ArrayRef<unsigned> CTAsPerCGA,
                                                 ArrayRef<unsigned> CTASplitNum,
                                                 ArrayRef<unsigned> CTAOrder) {
  int rank = CTAOrder.size();
  auto outDimNames = standardOutDimNames(ctx, rank);
  StringAttr kBlock = StringAttr::get(ctx, "block");

  LinearLayout layout = LinearLayout::empty();
  SmallVector<unsigned> splitNums(CTASplitNum.begin(), CTASplitNum.end());
  SmallVector<unsigned> ctas(CTAsPerCGA.begin(), CTAsPerCGA.end());

  for (int i = 0; i < rank; ++i) {
    int dim = CTAOrder[i];
    unsigned split = splitNums[dim];
    unsigned total = ctas[dim];
    assert(total % split == 0 && "invalid CTA encoding parameters");
    layout *= LinearLayout::identity1D(split, kBlock, outDimNames[dim]) *
              LinearLayout::zeros1D(total / split, kBlock, outDimNames[dim]);
  }

  layout = layout.transposeOuts(outDimNames);
  return CTAEncodingAttr::get(ctx, layout);
}

CTAEncodingAttr getCTALayout(Attribute layout) {
  if (auto ttgLayout = mlir::dyn_cast<LayoutEncodingTrait>(layout))
    return ttgLayout.getCTALayout();
  llvm::report_fatal_error("Unimplemented usage of getCTALayout");
  return {};
}

SmallVector<unsigned> CTAEncodingAttr::getCTAsPerCGA() const {
  auto ll = getLinearLayout();
  auto rank = ll.getNumOutDims();
  return basesPerDimImpl(ll.getBases(), StringAttr::get(getContext(), "block"),
                         rank, /*skipBroadcast=*/false);
}

SmallVector<unsigned> CTAEncodingAttr::getCTASplitNum() const {
  auto ll = getLinearLayout();
  auto rank = ll.getNumOutDims();
  return basesPerDimImpl(ll.getBases(), StringAttr::get(getContext(), "block"),
                         rank);
}

SmallVector<unsigned> getCTASplitNum(Attribute layout) {
  SmallVector<unsigned> res;
  if (auto ttgLayout = mlir::dyn_cast<LayoutEncodingTrait>(layout)) {
    return ttgLayout.getCTALayout().getCTASplitNum();
  // } else if (auto tmemLayout =
  //                mlir::dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
  //                    layout)) {
  //   res.resize(2);
  //   res[0] = tmemLayout.getCTASplitM();
  //   res[1] = tmemLayout.getCTASplitN();
  // } else if (auto tmemScaleLayout = mlir::dyn_cast<
  //                triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(layout)) {
  //   res.resize(2);
  //   res[0] = tmemScaleLayout.getCTASplitM();
  //   res[1] = tmemScaleLayout.getCTASplitN();
  } else {
    assert(false && "Unimplemented usage of getCTASplitNum");
  }
  return res;
}

SmallVector<unsigned> CTAEncodingAttr::getCTAOrder() const {
  auto rank = getRank();
  SmallVector<unsigned> defaultOrder(rank);
  std::iota(defaultOrder.begin(), defaultOrder.end(), 0);
  return orderPerDimImpl(getLinearLayout(),
                         StringAttr::get(getContext(), "block"), defaultOrder);
}

SmallVector<int64_t> getShapePerCTA(ArrayRef<unsigned> CTASplitNum,
                                    ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  auto splitNum = llvm::to_vector(CTASplitNum);
  if (splitNum.size() <= rank) { // pipelining
    splitNum.insert(splitNum.begin(), rank - splitNum.size(), 1);
  } else { // memory slicing
    splitNum =
        llvm::to_vector(llvm::drop_begin(splitNum, splitNum.size() - rank));
  }
  SmallVector<int64_t> shapePerCTA(rank);
  for (unsigned i = 0; i < rank; ++i) {
    shapePerCTA[i] = shape[i] / std::min<unsigned>(shape[i], splitNum[i]);
  }
  return shapePerCTA;
}

SmallVector<int64_t> getShapePerCTA(Attribute layout, ArrayRef<int64_t> shape) {
  return getShapePerCTA(getCTASplitNum(layout), shape);
}

LogicalResult BlockedEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> threadsPerWarp,
    ArrayRef<unsigned> warpsPerCTA, ArrayRef<unsigned> order,
    CTAEncodingAttr CTALayout) {
  if (!llvm::all_equal({sizePerThread.size(), threadsPerWarp.size(),
                        warpsPerCTA.size(), order.size()})) {
    return emitError() << "sizePerThread, threadsPerWarp, warpsPerCTA, and "
                          "order must all have the same rank.";
  }
  if (llvm::any_of(sizePerThread,
                   [](unsigned x) { return !llvm::isPowerOf2_64(x); })) {
    return emitError()
           << "Every element in sizePerThread must be a power of two.";
  }
  if (llvm::any_of(threadsPerWarp,
                   [](unsigned x) { return !llvm::isPowerOf2_64(x); })) {
    return emitError()
           << "Every element in threadsPerWarp must be a power of two.";
  }
  if (llvm::any_of(warpsPerCTA,
                   [](unsigned x) { return !llvm::isPowerOf2_64(x); })) {
    return emitError()
           << "Every element in warpsPerCTA must be a power of two.";
  }

  // Empty CTALayout is allowed, but if it's present its rank must match the
  // BlockedEncodingAttr's rank.
  if (order.size() != CTALayout.getRank()) {
    return emitError() << "BlockedEncodingAttr and CTALayout's fields must "
                          "have the same rank.";
  }
  return verifyLayoutOrder(emitError, order);
}

// 1 element per thread
// order = reverse(arange(rank))
triton::gpu::BlockedEncodingAttr
getDefaultBlockedEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                          int numWarps, int threadsPerWarp, int numCTAs) {
  int rank = shape.size();
  llvm::SmallVector<unsigned> order(rank);
  std::iota(order.begin(), order.end(), 0);
  std::reverse(order.begin(), order.end());
  llvm::SmallVector<unsigned> sizePerThread(rank, 1);
  triton::gpu::BlockedEncodingAttr encoding =
      triton::gpu::BlockedEncodingAttr::get(context, shape, sizePerThread,
                                            order, numWarps, threadsPerWarp,
                                            numCTAs);
  return encoding;
}

} // namespace gpu
} // namespace triton
} // namespace mlir

static LogicalResult parseIntAttrValue(AsmParser &parser, Attribute attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
  if (!intAttr) {
    parser.emitError(parser.getNameLoc(), "expected an integer type in ")
        << desc;
    return failure();
  }
  if (intAttr.getType().isSignedInteger()) {
    int64_t attrVal = intAttr.getSInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else if (intAttr.getType().isSignlessInteger()) {
    int64_t attrVal = intAttr.getInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else {
    value = intAttr.getUInt();
  }
  return success();
}

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned> &res,
                                       StringRef desc) {
  auto arrayAttr = mlir::dyn_cast<ArrayAttr>(attr.getValue());
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute i : arrayAttr) {
    unsigned value;
    if (parseIntAttrValue(parser, i, value, desc).failed())
      return failure();
    res.push_back(value);
  }
  return success();
};

static LogicalResult parseUInt(AsmParser &parser, const NamedAttribute &attr,
                               unsigned &value, StringRef desc) {
  return parseIntAttrValue(parser, attr.getValue(), value, desc);
};

static LogicalResult parseBoolAttrValue(AsmParser &parser, Attribute attr,
                                        bool &value, StringRef desc) {
  auto boolAttr = mlir::dyn_cast<BoolAttr>(attr);
  if (!boolAttr) {
    parser.emitError(parser.getNameLoc(), "expected a bool type in ") << desc;
    return failure();
  }
  value = boolAttr.getValue();
  return success();
}

static LogicalResult parseBool(AsmParser &parser, const NamedAttribute &attr,
                               bool &value, StringRef desc) {
  return parseBoolAttrValue(parser, attr.getValue(), value, desc);
};

static LogicalResult parseType(AsmParser &parser, const NamedAttribute &attr,
                               Type &value, StringRef desc) {
  auto typeAttr = mlir::dyn_cast<TypeAttr>(attr.getValue());
  if (!typeAttr) {
    parser.emitError(parser.getNameLoc(), "expected a Type in ") << desc;
    return failure();
  }
  value = typeAttr.getValue();
  return success();
}

static SmallVector<unsigned>
basesPerDimImpl(const LinearLayout::BasesT &namedBases, StringAttr dimName,
                size_t rank, bool skipBroadcast) {
  const auto &bases = namedBases.find(dimName)->second;

  if (bases.empty()) {
    return SmallVector<unsigned>(rank, 1);
  }

  SmallVector<unsigned> ret(rank, 1);
  auto nonZero = [](auto val) { return val != 0; };
  int nonZeroIdx = 0;
  for (const auto &basis : bases) {
    auto it = std::find_if(basis.begin(), basis.end(), nonZero);
    // Bases can have one or zero non-zero elements
    // Skip a basis if it's broadcasting (all zeros)
    // e.g. warps for DotOperandEncodingAttr (see ampereDotToLinearLayout)
    if (it != basis.end()) {
      nonZeroIdx = it - basis.begin();
      ret[nonZeroIdx] *= 2;
    } else if (!skipBroadcast) {
      // If we've seen a non-zero basis, we double the size of the previous dim
      // This is just needed to count the CTAsPerCGA
      ret[nonZeroIdx] *= 2;
    }
  }
  return ret;
}

void TritonGPUDialect::initialize() {
  registerTypes();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonGPU/IR/AttrDefs.cpp.inc"
      >();
//   addOperations<
// #define GET_OP_LIST
// #include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"
// #include "triton/Dialect/TritonGPU/IR/OpsEnums.cpp.inc"
//       >();
  // addInterfaces<TritonInlinerInterface>();
  // addInterfaces<TritonGPUOpAsmInterface>();
  // addInterfaces<TritonGPUInferLayoutInterface>();
  // addInterfaces<TritonGPUVerifyTensorLayoutInterface>();

  // RankedTensorType::attachInterface<TensorModel>(*getContext());
  // MemDescType::attachInterface<MemDescModel>(*getContext());
  // addAttributes<mlir::triton::gpu::CTAEncodingAttr>();
  // addAttributes<mlir::triton::gpu::BlockedEncodingAttr>();
}

LogicalResult TritonGPUDialect::verifyOperationAttribute(Operation *op,
                                                         NamedAttribute attr) {
  // Verify that dialect attributes are attached to the right ops.
  if (llvm::is_contained(
          {AttrNumCTAsName, AttrTargetName, AttrNumThreadsPerWarp},
          attr.getName()) &&
      !isa<ModuleOp>(op)) {
    return op->emitOpError("has unexpected attribute ")
           << attr.getName() << " which is expected only on `module` ops";
  }
  if (attr.getName() == AttrNumWarpsName && !isa<ModuleOp, func::FuncOp>(op)) {
    return op->emitOpError("has unexpected attribute ")
           << attr.getName()
           << " which is expected only on `module` or `tt.func` ops";
  }

  // Verify that all ops in a tt.warp_specialize op have partition ids
  if (attr.getName() == "tt.warp_specialize") {
    if (!isa<scf::ForOp>(op)) {
      return op->emitOpError("has unexpected attribute ")
             << attr.getName() << " which is expected only on `scf.for` ops";
    }
    Operation *failedOp = nullptr;
    op->walk([&](Operation *childOp) {
      if (!childOp->hasAttr(kPartitionAttrName)) {
        failedOp = childOp;
        WalkResult::interrupt();
      }
    });
    if (failedOp) {
      return failedOp->emitOpError("does not have expected attribute ")
             << kPartitionAttrName
             << " which is expected on all child ops of an op with "
                "attribute `tt.warp_specialize`";
    }
  }

  // Verify that partition id lists are non-empty, sorted and have no duplicates
  auto verifyPartitionIds =
      [&](const ArrayRef<int> &partitionIds) -> LogicalResult {
    SetVector<int> idSet;
    for (auto id : partitionIds) {
      if (idSet.contains(id))
        return op->emitOpError("has duplicated partition ids in attribute ")
               << attr.getName();
      idSet.insert(id);
    }
    if (idSet.empty())
      return op->emitOpError("has no partition ids in attribute ")
             << attr.getName();
    auto ids = idSet.takeVector();
    SmallVector<int> sortedIds(ids.begin(), ids.end());
    std::sort(sortedIds.begin(), sortedIds.end());
    if (ids != sortedIds)
      return op->emitOpError("partition ids not in sorted order in attribute ")
             << attr.getName();
    return success();
  };

  if (attr.getName() == kPartitionAttrName) {
    auto result = verifyPartitionIds(
        cast<DenseI32ArrayAttr>(attr.getValue()).asArrayRef());
    if (failed(result))
      return result;
  }
  if (attr.getName() == kPartitionOutputsAttrName) {
    auto arrayAttr = cast<ArrayAttr>(attr.getValue());
    for (auto idx = 0; idx < arrayAttr.size(); idx++) {
      auto result = verifyPartitionIds(
          cast<DenseI32ArrayAttr>(arrayAttr[idx]).asArrayRef());
      if (failed(result))
        return result;
    }
  }

  // // Verify that op partitions include partitions of all child ops
  // if (attr.getName() == kPartitionAttrName && op->getNumRegions() != 0) {
  //   SetVector<int> expectedIds;
  //   for (auto &region : op->getRegions()) {
  //     for (auto &block : region.getBlocks()) {
  //       for (auto &childOp : block.getOperations()) {
  //         if (isa<scf::YieldOp, ub::PoisonOp>(childOp)) {
  //           // yield ops and ub.poison do not need partition ids
  //           continue;
  //         }
  //         if (!childOp.hasAttr(kPartitionAttrName))
  //           return childOp.emitOpError("does not have expected attribute ")
  //                  << kPartitionAttrName
  //                  << " which is expected for ops whose parent has partitions";
  //         auto ids = getPartitionIds(&childOp);
  //         expectedIds.insert(ids.begin(), ids.end());
  //       }
  //     }
  //   }
  //   auto partitionIds = getPartitionIds(op);
  //   for (auto id : expectedIds) {
  //     if (!partitionIds.contains(id)) {
  //       return op->emitOpError("partition ids in attr ")
  //              << attr.getName()
  //              << " does not contain partition ids of all child ops";
  //     }
  //   }
  // }

  // if (attr.getName() == kPartitionOutputsAttrName) {
  //   if (!isa<scf::ForOp, scf::IfOp, triton::ReduceOp>(op))
  //     return op->emitOpError("has unexpected attribute ") << attr.getName();

  //   // Verify that number of output partitions matches number of For/If results
  //   size_t numResults = 0;
  //   if (isa<scf::ForOp>(op)) {
  //     numResults = cast<scf::ForOp>(op).getResults().size();
  //   } else if (isa<scf::IfOp>(op)) {
  //     numResults = cast<scf::IfOp>(op).getResults().size();
  //   } else {
  //     numResults = cast<triton::ReduceOp>(op).getResults().size();
  //   }

  //   if (cast<ArrayAttr>(attr.getValue()).size() != numResults) {
  //     return op->emitOpError("does not have expected number of output "
  //                            "partition sets in attr ")
  //            << attr.getName() << "; should match number of results";
  //   }

  //   // Verify that union of op output partitions is a subset of op partitions
  //   if (!op->hasAttr(kPartitionAttrName))
  //     return op->emitOpError("does not have expected attribute ")
  //            << kPartitionAttrName << " which is expected for ops with attr "
  //            << kPartitionOutputsAttrName;
  //   auto partitionIds = getPartitionIds(op);

  //   SetVector<int> outputPartitionIdsUnion;
  //   for (auto outputPartitionIds : getPartitionOutputs(op)) {
  //     outputPartitionIdsUnion.insert(outputPartitionIds.begin(),
  //                                    outputPartitionIds.end());
  //   }
  //   if (!std::all_of(outputPartitionIdsUnion.begin(),
  //                    outputPartitionIdsUnion.end(),
  //                    [&](int id) { return partitionIds.contains(id); })) {
  //     return op->emitOpError("partition ids in attr ")
  //            << kPartitionAttrName
  //            << " must be the union of all partition ids in " << attr.getName();
  //   }
  // }

  return success();
}

// Print the CTA encoding as `CGALayout = [[...]]` when the layout is
// non-trivial.
static void maybePrintCTALayout(mlir::MLIRContext *context,
                                mlir::AsmPrinter &printer,
                                CTAEncodingAttr layout, unsigned rank) {
  if (layout == CTAEncodingAttr::getDefault(context, rank))
    return;

  auto kBlock = StringAttr::get(context, "block");
  const auto &basesMap = layout.getLinearLayout().getBases();
  auto it = basesMap.find(kBlock);
  assert(it != basesMap.end());
  const auto &bases = it->second;
  // This is the default layout
  assert(!bases.empty());

  printer << ", CGALayout = [";
  llvm::interleaveComma(bases, printer, [&](const std::vector<int32_t> &vec) {
    printer << "[";
    llvm::interleaveComma(vec, printer);
    printer << "]";
  });
  printer << "]";
}

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonGPU/IR/AttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/AttrDefs.cpp.inc"
#undef GET_ATTRDEF_CLASSES


CTAEncodingAttr linearToCTAEncodingAttr(const LinearLayout &ll,
                                        ArrayRef<unsigned> cgaLogicalShape) {
  // Compute the shapePerCTA
  auto shape = ll.getOutDims();
  for (int i = 0; i < shape.size(); ++i) {
    shape[i].second /= cgaLogicalShape[i];
  }
  auto inDims = to_vector(ll.getInDimNames());
  auto kBlock = inDims.back();
  assert(kBlock.str() == "block");
  inDims.pop_back();
  auto outDims = to_vector(ll.getOutDimNames());
  auto subLl = ll.sublayout(inDims, outDims);
  // sublayout returns the same output size. We trim it to the
  // real size
  subLl = LinearLayout(subLl.getBases(), shape, false);
  // The ctaLayout is what we get after dividing on the left by
  // the layout in a single CTA
  auto maybeCtaLayout = divideLeft(ll, subLl);
  assert(maybeCtaLayout.has_value());
  auto *ctx = inDims[0].getContext();
  auto ctaLayout = maybeCtaLayout->sublayout({kBlock}, outDims);
  return CTAEncodingAttr::get(ctx, std::move(ctaLayout));
}

SmallVector<unsigned>
LinearEncodingAttr::orderPerDim(StringAttr dimName,
                                ArrayRef<unsigned> defaultOrder) const {
  return orderPerDimImpl(getLinearLayout(), dimName, defaultOrder);
}

// [Note. Divergence of methods wrt. legacy layouts]
// For smaller shapes where the CTATile is larger than the output
// tensor, some methods return different values than the legacy layouts. I think
// this is benign tho. An example: what is the vector of `warpsPerCTA` if
// all the warps hold the same data? I think it should be [1, 1], even if we
// have 4 warps. But perhaps for this we have to add some masking in some
// places... We'll see
SmallVector<unsigned> LinearEncodingAttr::getRepOrder() const {
  // This is not correct, but:
  // - It happens to agree in most places with the legacy layout
  // - getRepOrder does not make sense for LinearEncodingAttr as it already has
  //   the same shape as the tensor that uses it
  return getOrder();
}

CTAEncodingAttr LinearEncodingAttr::getCTALayout() const {
  auto splitNum = basesPerDim(StringAttr::get(getContext(), "block"));
  return linearToCTAEncodingAttr(getLinearLayout(), splitNum);
}
SmallVector<unsigned> LinearEncodingAttr::getWarpsPerCTA() const {
  return basesPerDim(StringAttr::get(getContext(), "warp"));
}
SmallVector<unsigned> LinearEncodingAttr::getWarpOrder() const {
  return orderPerDim(StringAttr::get(getContext(), "warp"), getOrder());
}
SmallVector<unsigned> LinearEncodingAttr::getThreadsPerWarp() const {
  return basesPerDim(StringAttr::get(getContext(), "lane"));
}
SmallVector<unsigned> LinearEncodingAttr::getThreadOrder() const {
  return orderPerDim(StringAttr::get(getContext(), "lane"), getOrder());
}

SmallVector<unsigned> LinearEncodingAttr::getSizePerThread() const {
  auto rank = getOrder().size();
  auto ll = getLinearLayout();
  auto ctx = getContext();
  auto kRegister = StringAttr::get(ctx, "register");
  auto splitNum = getCTALayout().getCTASplitNum();

  // We canonicalize on the spot, as if we use CGAs the regs are not in
  // canonical form The order is [reg, lane, warp, rep, block], so we first
  // remove the blocks
  llvm::SmallVector<unsigned> ctaShape;
  for (auto [shape, cgaNum] : llvm::zip(ll.getOutDimSizes(), splitNum)) {
    ctaShape.push_back(shape / cgaNum);
  }
  LinearLayout::BasesT bases = ll.getBases();

  llvm::SetVector<unsigned> reverseRepOrder;
  auto nonZero = [](auto val) { return val != 0; };
  auto &registers = bases[kRegister];
  while (!registers.empty()) {
    auto &basis = registers.back();
    auto it = std::find_if(basis.begin(), basis.end(), nonZero);
    // If there's broadcasting (base == zeros) there are no more reps
    if (it == basis.end()) {
      break;
    }
    auto dim = it - basis.begin();
    reverseRepOrder.insert(dim);
    // As soon as we stop finding reps, we stop
    if (dim != reverseRepOrder.back() || 2 * basis[dim] != ctaShape[dim]) {
      break;
    }
    ctaShape[dim] /= 2;
    registers.pop_back();
  }
  return basesPerDimImpl(bases, kRegister, rank);
}

SmallVector<unsigned> LinearEncodingAttr::getOrder() const {
  auto rank = getLinearLayout().getNumOutDims();
  SmallVector<unsigned> order(rank);
  // Choose [rank-1, rank-2, ... 0] as the default order in case
  // there are dims that do not move in the register
  // This order is as good as any really
  std::iota(order.rbegin(), order.rend(), 0);

  return orderPerDim(StringAttr::get(getContext(), "register"), order);
}

LinearLayout LinearEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto ll = getLinearLayout();
  auto canonicalDims = llvm::to_vector(ll.getOutDimNames());
  llvm::SmallDenseMap<StringAttr, int64_t> namedShape;
  llvm::SmallVector<StringAttr> permutedDims;
  for (auto dim : getRepOrder()) {
    permutedDims.push_back(canonicalDims[dim]);
    namedShape[canonicalDims[dim]] = shape[dim];
  }
  ll = ll.transposeOuts(permutedDims);
  ll = ensureLayoutNotSmallerThan(ll, namedShape);
  ll = ensureLayoutNotLargerThan(ll, namedShape, /*broadcastRegisters=*/false);
  ll = ll.transposeOuts(canonicalDims);
  return ll;
}

SmallVector<unsigned>
LinearEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape) const {
  // When broadcasting the layout the shape changes, otherwise the shape is
  // the same as the shape of the tensor
  // We can either have BroadcastOp with SameOperandsAndResultEncoding, or keep
  // the invariant that the shape of the LL is that of the tensor
  // We choose the former for BC
  auto scaledLayout = get(getContext(), toLinearLayout(shape));
  auto kRegister = StringAttr::get(getContext(), "register");
  return scaledLayout.basesPerDim(kRegister, /*skipBroadcast=*/false);
}

SmallVector<unsigned>
LinearEncodingAttr::getContig(const char *inDim,
                              SmallVector<unsigned int> lowerContig) const {
  auto ll = getLinearLayout();
  const auto &bases =
      ll.getBases().find(StringAttr::get(getContext(), inDim))->second;
  auto order = getOrder();
  auto rank = order.size();

  SmallVector<unsigned> contig(lowerContig);
  auto basisIt = bases.begin();
  for (unsigned dim : order) {
    std::vector<int32_t> basis(rank, 0);
    basis[dim] = contig[dim];

    while (basisIt != bases.end() && *basisIt == basis) {
      contig[dim] *= 2;
      basis[dim] *= 2;
      ++basisIt;
    }
  }
  return contig;
}

SmallVector<unsigned> LinearEncodingAttr::getContigPerThread() const {
  SmallVector<unsigned> contig(getOrder().size(), 1);
  return getContig("register", contig);
}

SmallVector<unsigned> LinearEncodingAttr::getContigPerWarp() const {
  return getContig("lane", getContigPerThread());
}

unsigned
LinearEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape) const {
  return product(getElemsPerThread(shape));
}


// We don't use the default implementation as it's a bit too verbose
// This prints in the following format that is shape agnostic, in the sense
// that we don't print explicitly the outShape of the LL
// We always assume LLs to be surjective
// <{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
//   lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
//   warp = [[16, 0], [32, 0]],
//   block = []}>
static void printLinearLayout(AsmPrinter &printer, const LinearLayout &ll) {
  printer << join(ll.getBases(), ", ", [](const auto &base) {
    return base.first.str() + " = " + "[" +
           join(base.second, ", ",
                [](const std::vector<int32_t> &vec) {
                  return "[" + join(vec, ", ") + "]";
                }) +
           "]";
  });
}

std::optional<LinearLayout>
parseLinearLayout(const DictionaryAttr &dict, AsmParser &parser,
                  ArrayRef<std::string> inDimNames) {
  LinearLayout::BasesT bases;

  // Parse the basis names in order (the order is relevant)
  for (const auto &inDimNameStr : inDimNames) {
    auto inDimName = StringAttr::get(parser.getContext(), inDimNameStr);
    Attribute value = dict.get(inDimName);
    if (!value) {
      parser.emitError(parser.getCurrentLocation(), "Expected basis of '")
          << inDimName.getValue() << "' not found";
      return {};
    }
    // Expecting an array of arrays
    auto arrayOfArraysAttr = mlir::dyn_cast<ArrayAttr>(value);
    if (!arrayOfArraysAttr) {
      parser.emitError(parser.getCurrentLocation(),
                       "Expected array of arrays for basis of '")
          << inDimName.getValue() << "'";
      return {};
    }

    std::vector<std::vector<int32_t>> inDimBases;
    for (Attribute arrayAttr : arrayOfArraysAttr) {
      auto intArrayAttr = mlir::dyn_cast<ArrayAttr>(arrayAttr);
      if (!intArrayAttr) {
        parser.emitError(parser.getCurrentLocation(),
                         "Expected array of integers in basis for '")
            << inDimName.getValue() << "'";
        return {};
      }
      std::vector<int32_t> basis;
      for (Attribute intAttr : intArrayAttr) {
        auto intValueAttr = mlir::dyn_cast<IntegerAttr>(intAttr);
        if (!intValueAttr) {
          parser.emitError(parser.getCurrentLocation(),
                           "Expected integer in basis for '")
              << inDimName.getValue() << "'";
          return {};
        }
        basis.push_back(intValueAttr.getInt());
      }
      inDimBases.push_back(std::move(basis));
    }
    bases[inDimName] = std::move(inDimBases);
  }
  size_t rank = 0;
  for (const auto &basesDim : llvm::make_second_range(bases)) {
    if (!basesDim.empty()) {
      rank = basesDim[0].size();
      break;
    }
  }

  // To implement this we'd need to serialise the rank as well.
  // We can do this if we ever need it
  if (rank == 0) {
    parser.emitError(parser.getCurrentLocation(), "Empty Layout not supported");
    return {};
  }

  // Generate standared outDimNames (dim0, dim1, ...)
  SmallVector<StringAttr> outDimNames;
  for (int i = 0; i < rank; ++i) {
    outDimNames.push_back(
        StringAttr::get(parser.getContext(), "dim" + llvm::Twine(i)));
  }

  // Create LinearLayout
  return LinearLayout(std::move(bases), std::move(outDimNames));
}


//===----------------------------------------------------------------------===//
// Linear Encoding
//===----------------------------------------------------------------------===//

void LinearEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{";
  printLinearLayout(printer, getLinearLayout());
  printer << "}>";
}

Attribute LinearEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};

  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};

  if (parser.parseGreater().failed())
    return {};

  std::vector<std::string> inDimNames = {"register", "lane", "warp", "block"};
  auto maybeLL = parseLinearLayout(dict, parser, inDimNames);
  if (!maybeLL.has_value())
    return {};

  // Create and return the LinearEncodingAttr
  return parser.getChecked<LinearEncodingAttr>(parser.getContext(),
                                               std::move(*maybeLL));
}


//===----------------------------------------------------------------------===//
// Blocked Encoding
//===----------------------------------------------------------------------===//

std::optional<CTAEncodingAttr> parseCTAAttr(AsmParser &parser, Attribute attr,
                                            unsigned rank) {
  if (!attr)
    return CTAEncodingAttr::getDefault(parser.getContext(), rank);

  auto array = llvm::dyn_cast<ArrayAttr>(attr);
  if (!array) {
    parser.emitError(parser.getNameLoc(),
                     "expected array value for 'CGALayout'");
    return {};
  }

  auto ctx = parser.getContext();
  auto cgaName = StringAttr::get(ctx, "CGALayout");
  std::vector<std::vector<int32_t>> bases;
  bases.reserve(array.size());
  for (Attribute vecAttr : array) {
    SmallVector<unsigned> basisValues;
    NamedAttribute basisAttr(cgaName, vecAttr);
    if (parseIntArrayAttr(parser, basisAttr, basisValues, "CGALayout entry")
            .failed())
      return {};
    if (basisValues.size() != rank) {
      parser.emitError(parser.getNameLoc())
          << "'CGALayout' entry length does not match rank " << rank;
      return {};
    }
    std::vector<int32_t> basis;
    basis.reserve(basisValues.size());
    for (unsigned value : basisValues)
      basis.push_back(static_cast<int32_t>(value));
    bases.push_back(std::move(basis));
  }

  LinearLayout::BasesT namedBases;
  namedBases.insert(
      std::make_pair(StringAttr::get(ctx, "block"), std::move(bases)));
  LinearLayout ll(namedBases, standardOutDimNames(ctx, rank));
  return CTAEncodingAttr::get(ctx, std::move(ll));
}

Attribute BlockedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned> sizePerThread;
  SmallVector<unsigned> threadsPerWarp;
  SmallVector<unsigned> warpsPerCTA;
  SmallVector<unsigned> order;
  Attribute ctaAttr = nullptr;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "sizePerThread") {
      if (parseIntArrayAttr(parser, attr, sizePerThread,
                            "number of elements per thread")
              .failed())
        return {};
    } else if (attr.getName() == "threadsPerWarp") {
      if (parseIntArrayAttr(parser, attr, threadsPerWarp,
                            "number of threads per warp")
              .failed())
        return {};
    } else if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA,
                            "number of warps per CTA")
              .failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else if (attr.getName() == "CGALayout") {
      ctaAttr = attr.getValue();
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  std::optional<CTAEncodingAttr> CTALayout =
      parseCTAAttr(parser, ctaAttr, /*rank=*/sizePerThread.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<BlockedEncodingAttr>(parser.getContext(),
                                                sizePerThread, threadsPerWarp,
                                                warpsPerCTA, order, *CTALayout);
}

void BlockedEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "sizePerThread = [" << ArrayRef(getSizePerThread()) << "]"
          << ", threadsPerWarp = [" << ArrayRef(getThreadsPerWarp()) << "]"
          << ", warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]"
          << ", order = [" << getOrder() << "]";

  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getSizePerThread().size());

  printer << "}>";
}

// FIXME Can we take the LinearLayout by const&?
LogicalResult
LinearEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           LinearLayout linearLayout) {
  // Example of LinearEncodingAttr
  // <{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
  //   lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
  //   warp = [[16, 0], [32, 0]],
  //   block = []}>
  // The input dims must be {register, lane, warp, block}
  // The output dims of the linear layout should be dim0..dim[rank-1]

  static const auto expectedInDims =
      SmallVector<std::string>({"register", "lane", "warp", "block"});
  for (const auto &[i, dims] : llvm::enumerate(
           llvm::zip(linearLayout.getInDimNames(), expectedInDims))) {
    const auto &[dim, expectedDimStr] = dims;
    if (dim.str() != expectedDimStr) {
      return emitError() << "Expected input dimension " << i << " to be '"
                         << expectedDimStr << "'. Got " << dim;
    }
  }

  // outDims are ['dim0', 'dim1', ...]
  for (auto [i, dim] : llvm::enumerate(linearLayout.getOutDimNames())) {
    if (dim.str() != ("dim" + llvm::Twine(i)).str()) {
      return emitError()
             << "Expected output dimensions to be ['dim0', 'dim1', ...]. Got "
             << dim << " at position " << i;
    }
  }

  const auto &bases = linearLayout.getBases();
  auto nonZero = [](auto val) { return val != 0; };
  for (const auto &dimBases : llvm::make_second_range(bases)) {
    if (!llvm::all_of(dimBases, [&](const auto &basis) {
          return std::count_if(basis.begin(), basis.end(), nonZero) <= 1;
        })) {
      return emitError()
             << "In a distributed layout, each base must move in at most one "
                "dimension.";
    }
  }

  return success();
}

// If we only had BlockedEncodingAttr, we could simply return ArrayRefs here.
// But we need to have a consistent interface with e.g. SliceEncodingAttr, which
// computes some of these fields.
SmallVector<unsigned> BlockedEncodingAttr::getRepOrder() const {
  return SmallVector<unsigned>(getOrder());
}

template <typename SpecificEncoding>
Attribute parseSwizzledEncoding(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned vec = 0;
  unsigned perPhase = 0;
  unsigned maxPhase = 0;
  SmallVector<unsigned> order;
  Attribute ctaAttr = nullptr;
  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "vec") {
      if (parseUInt(parser, attr, vec, "vec").failed())
        return {};
    } else if (attr.getName() == "perPhase") {
      if (parseUInt(parser, attr, perPhase, "perPhase").failed())
        return {};
    } else if (attr.getName() == "maxPhase") {
      if (parseUInt(parser, attr, maxPhase, "maxPhase").failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else {
      if (attr.getName() == "CGALayout") {
        ctaAttr = attr.getValue();
      } else {
        parser.emitError(parser.getNameLoc(), "unexpected key: ")
            << attr.getName().strref();
        return {};
      }
    }
  }

  if (auto CTALayout = parseCTAAttr(parser, ctaAttr, order.size()))
    return parser.getChecked<SpecificEncoding>(
        parser.getContext(), vec, perPhase, maxPhase, order, *CTALayout);
  return {};
}


//===----------------------------------------------------------------------===//
// SwizzledShared encoding
//===----------------------------------------------------------------------===//

LogicalResult
SwizzledSharedEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   unsigned vec, unsigned perPhase,
                                   unsigned maxPhase, ArrayRef<unsigned> order,
                                   CTAEncodingAttr ctaLayout) {
  if (order.size() != ctaLayout.getRank()) {
    return emitError() << "order size (" << order.size()
                       << ") must match CTALayout rank (" << ctaLayout.getRank()
                       << ")";
  }
  return verifyLayoutOrder(emitError, order);
}

Attribute SwizzledSharedEncodingAttr::parse(AsmParser &parser, Type type) {
  return parseSwizzledEncoding<SwizzledSharedEncodingAttr>(parser, type);
}

void SwizzledSharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "vec = " << getVec() //
          << ", perPhase = " << getPerPhase()
          << ", maxPhase = " << getMaxPhase() //
          << ", order = [" << getOrder() << "]";
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getOrder().size());
  printer << "}>";
}


//===----------------------------------------------------------------------===//
// SharedLinear encoding
//===----------------------------------------------------------------------===//

LogicalResult
SharedLinearEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 LinearLayout linearLayout,
                                 unsigned layoutAlignment) {
  if (layoutAlignment == 0 || !llvm::isPowerOf2_32(layoutAlignment)) {
    return emitError() << "alignment must be a positive power of two";
  }
  static const auto expectedInDims =
      SmallVector<std::string>({"offset", "block"});
  for (const auto &[index, dims] : llvm::enumerate(
           llvm::zip(linearLayout.getInDimNames(), expectedInDims))) {
    const auto &[dim, expected] = dims;
    if (dim.str() != expected) {
      return emitError() << "Expected input dimension " << index << " to be '"
                         << expected << "'. Got " << dim;
    }
  }

  for (auto [i, dim] : llvm::enumerate(linearLayout.getOutDimNames())) {
    if (dim.str() != ("dim" + llvm::Twine(i)).str()) {
      return emitError()
             << "Expected output dimensions to be ['dim0', 'dim1', ...]. Got "
             << dim << " at position " << i;
    }
  }

  SmallVector<StringAttr> outDimNames =
      llvm::to_vector(linearLayout.getOutDimNames());
  if (outDimNames.empty()) {
    return emitError()
           << "SharedLinearEncodingAttr requires at least one output"
              " dimension.";
  }

  auto *ctx = outDimNames.front().getContext();
  auto kOffset = StringAttr::get(ctx, "offset");
  auto kBlock = StringAttr::get(ctx, "block");

  if (!linearLayout.isSurjective()) {
    return emitError() << "The layout must be surjective";
  }

  LinearLayout withoutBroadcast =
      linearLayout.removeZeroBasesAlongDim(kOffset).removeZeroBasesAlongDim(
          kBlock);
  if (!withoutBroadcast.isInvertible()) {
    return emitError()
           << "After removing the zero bases the layout must be bijective";
  }

  return success();
}

void SharedLinearEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{";
  auto layout = getLinearLayout();
  auto kBlock = StringAttr::get(getContext(), "block");
  auto kOffset = StringAttr::get(getContext(), "offset");
  if (layout.getBases().lookup(kBlock).empty()) {
    layout =
        layout.sublayout({kOffset}, llvm::to_vector(layout.getOutDimNames()));
  }
  printLinearLayout(printer, layout);
  printer << "}, alignment = " << getAlignment() << ">";
}

Attribute SharedLinearEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};

  DictionaryAttr layoutDictRaw;
  if (parser.parseAttribute(layoutDictRaw).failed())
    return {};

  if (layoutDictRaw.get("alignment")) {
    parser.emitError(parser.getCurrentLocation())
        << "alignment must be specified outside of the linear layout braces";
    return {};
  }

  NamedAttrList layoutAttrList(layoutDictRaw.getValue());
  auto *ctx = parser.getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  if (!layoutAttrList.get(kBlock)) {
    layoutAttrList.push_back({kBlock, ArrayAttr::get(ctx, {})});
  }

  DictionaryAttr layoutDict = layoutAttrList.getDictionary(ctx);

  // Parse alignment
  unsigned layoutAlignment;
  if (parser.parseComma().failed())
    return {};
  if (parser.parseKeyword("alignment").failed() || parser.parseEqual().failed())
    return {};
  if (parser.parseInteger(layoutAlignment).failed())
    return {};

  if (parser.parseGreater().failed())
    return {};

  std::vector<std::string> inDimNames = {"offset", "block"};
  auto maybeLL = parseLinearLayout(layoutDict, parser, inDimNames);
  if (!maybeLL.has_value())
    return {};

  // Special case for cleaner errors
  if (layoutDict.get("alignment")) {
    parser.emitError(parser.getCurrentLocation())
        << "alignment must be specified outside of the linear layout braces";
    return {};
  }

  if (layoutDict.size() != 2) {
    parser.emitError(parser.getCurrentLocation())
        << "SharedLinearEncodingAttr must have exactly two attributes: offset "
           "and block";
    return {};
  }

  return parser.getChecked<SharedLinearEncodingAttr>(
      parser.getContext(), std::move(*maybeLL), layoutAlignment);
}

SmallVector<unsigned>
SharedLinearEncodingAttr::basesPerDim(StringAttr dimName,
                                      bool skipBroadcast) const {
  auto ll = getLinearLayout();
  auto rank = ll.getNumOutDims();
  return basesPerDimImpl(ll.getBases(), dimName, rank, skipBroadcast);
}

SmallVector<unsigned>
SharedLinearEncodingAttr::orderPerDim(StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) const {
  return orderPerDimImpl(getLinearLayout(), dimName, defaultOrder);
}

SmallVector<unsigned> SharedLinearEncodingAttr::getOrder() const {
  auto ll = getLinearLayout();
  auto rank = ll.getNumOutDims();
  SmallVector<unsigned> defaultOrder(rank);
  std::iota(defaultOrder.rbegin(), defaultOrder.rend(), 0);
  return orderPerDim(StringAttr::get(getContext(), "offset"), defaultOrder);
}

CTAEncodingAttr SharedLinearEncodingAttr::getCTALayout() const {
  auto splitNum = basesPerDim(StringAttr::get(getContext(), "block"));
  return linearToCTAEncodingAttr(getLinearLayout(), splitNum);
}
LinearLayout
SharedLinearEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto ll = getLinearLayout();
  auto outDimNames = llvm::to_vector(ll.getOutDimNames());
  assert(shape.size() == outDimNames.size());
  // We don't support automatic broadcasting for shared linear layouts
  for (auto [size, llSize] : llvm::zip(shape, ll.getOutDimSizes())) {
    assert(size == llSize);
  }
  return ll;
}


//===----------------------------------------------------------------------===//
// PaddedShared encoding
//===----------------------------------------------------------------------===//

Attribute PaddedSharedEncodingAttr::parse(AsmParser &parser, Type type) {
  // <[
  if (failed(parser.parseLess()) || failed(parser.parseLSquare()))
    return {};

  // <interval_i>:+<padding_i>
  SmallVector<unsigned, 4> intervals, paddings;
  auto parseIntervalPaddingPair = [&]() {
    unsigned interval = 0, padding = 0;
    if (failed(parser.parseInteger(interval)) || failed(parser.parseColon()) ||
        failed(parser.parsePlus()) || failed(parser.parseInteger(padding)))
      return failure();
    intervals.push_back(interval);
    paddings.push_back(padding);
    return success();
  };
  // ]
  if (failed(parser.parseCommaSeparatedList(parseIntervalPaddingPair)) ||
      failed(parser.parseRSquare()))
    return {};

  // {<attr-dict>}
  auto attrList = DictionaryAttr::get(parser.getContext());
  if (failed(parser.parseAttribute(attrList)))
    return {};

  // We have 2 possible formats for the attr-dict:
  //  1) offset=[..], block=[..] handled by parseLinearLayout
  //  2) order=[..], shape=[..] which creates an identity mapping

  std::optional<LinearLayout> maybeLL;
  // Assume it's the first variant if offset or block is defined
  if (attrList.contains("offset") || attrList.contains("block")) {
    std::vector<std::string> inDimNames = {"offset", "block"};
    // Error out on additional attribute names
    for (const NamedAttribute &attr : attrList) {
      if (!llvm::is_contained(inDimNames, attr.getName())) {
        parser.emitError(parser.getCurrentLocation(), "Unexpected attribute ")
            << attr.getName() << " found";
      }
    }
    maybeLL = parseLinearLayout(attrList, parser, inDimNames);
  } else {
    // Parse the second form
    SmallVector<unsigned> order;
    SmallVector<unsigned> shape;
    for (const NamedAttribute &attr : attrList) {
      if (attr.getName() == "order") {
        if (parseIntArrayAttr(parser, attr, order, "order").failed())
          return {};
      } else if (attr.getName() == "shape") {
        if (parseIntArrayAttr(parser, attr, shape, "shape").failed())
          return {};
      } else {
        parser.emitError(parser.getCurrentLocation(), "Unexpected attribute ")
            << attr.getName() << " found";
        return {};
      }
    }

    if (order.size() != shape.size()) {
      parser.emitError(parser.getCurrentLocation(),
                       "Mismatch of shape and order ranks in padded layout");
      return {};
    }

    // Create identity mapping based on shape and order
    auto kOffset = StringAttr::get(parser.getContext(), "offset");
    maybeLL = identityStandardND(kOffset, shape, order);
    maybeLL = combineCtaCgaWithShape(
        *maybeLL,
        CTAEncodingAttr::getDefault(parser.getContext(), shape.size()),
        SmallVector<int64_t>(ArrayRef(shape)));
  }

  if (!maybeLL.has_value())
    return {};

  // >
  if (parser.parseGreater().failed())
    return {};

  return parser.getChecked<PaddedSharedEncodingAttr>(
      parser.getContext(), intervals, paddings, *maybeLL);
}

void PaddedSharedEncodingAttr::print(AsmPrinter &printer) const {

  auto *ctx = getContext();
  const auto &ll = getLinearComponent();

  printer << "<[";
  llvm::interleaveComma(llvm::zip(getIntervals(), getPaddings()), printer,
                        [&](std::tuple<unsigned, unsigned> intervalPad) {
                          printer << std::get<0>(intervalPad) << ":+"
                                  << std::get<1>(intervalPad);
                        });
  printer << "] {";

  // We have a short hand form if linearComponent:
  //  1) does have an empty CTA layout (empty block dim)
  //  2) offsets are an identity mapping
  auto kOffset = StringAttr::get(ctx, "offset");
  auto kBlock = StringAttr::get(ctx, "block");
  auto shape = SmallVector<unsigned>(ll.getOutDimSizes());

  bool hasEmptyBlock = ll.getInDimSizeLog2(kBlock) == 0;

  LinearLayout identity = identityStandardND(kOffset, shape, getOrder())
                              .transposeOuts(to_vector(ll.getOutDimNames()));
  auto offsetLayout = ll.sublayout({kOffset}, to_vector(ll.getOutDimNames()));

  if (hasEmptyBlock && offsetLayout == identity) {
    printer << "order = [" << ArrayRef(getOrder()) << "], shape = ["
            << ArrayRef(shape) << "]";
  } else {
    printLinearLayout(printer, getLinearComponent());
  }

  printer << "}>";
}

LogicalResult PaddedSharedEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<unsigned> intervals,
    ArrayRef<unsigned> paddings, LinearLayout linearComponent) {
  if (intervals.size() != paddings.size())
    return emitError() << "intervals size (" << intervals.size()
                       << ") must match paddings size (" << paddings.size()
                       << ")";

  if (intervals.empty())
    return emitError() << "must have at least one interval-padding pair";

  if (!llvm::all_of(intervals, llvm::isPowerOf2_32))
    return emitError() << "interval values must all be power of two";
  if (!llvm::all_of(paddings, llvm::isPowerOf2_32))
    return emitError() << "padding values must all be power of two";

  llvm::SmallSet<unsigned, 4> intervalValues(intervals.begin(),
                                             intervals.end());
  if (intervalValues.size() != intervals.size())
    return emitError() << "interval values cannot have duplicates";

  const auto &ll = linearComponent;
  // The linear layout should map from [offset, block] to [dim0..dimN). All
  // bases should be 0 or power of twos and move in a single direction without
  // broadcasting

  if (ll == LinearLayout::empty())
    return emitError() << "linearComponent cannot be empty";

  assert(!ll.getInDimNames().empty());
  auto *ctx = ll.getInDimNames().begin()->getContext();

  if (!llvm::equal(ll.getInDimNames(),
                   std::array{StringAttr::get(ctx, "offset"),
                              StringAttr::get(ctx, "block")})) {
    return emitError()
           << "linearComponent must have [offset, block] as input dims";
  }

  if (!llvm::equal(ll.getOutDimNames(),
                   standardOutDimNames(ctx, ll.getNumOutDims()))) {
    return emitError()
           << "Expected output dimensions to be ['dim0', 'dim1', ...].";
  }

  const auto &bases = ll.getBases();

  // Check that we are not broadcasting or having repeated bases
  if (!ll.isInvertible()) {
    return emitError() << "Broadcasting is not supported.";
  }

  auto nonZero = [](auto val) { return val != 0; };
  for (const auto &dimBases : llvm::make_second_range(bases)) {
    if (!llvm::all_of(dimBases, [&](const auto &basis) {
          return llvm::count_if(basis, nonZero) <= 1;
        })) {
      return emitError()
             << "Each offset basis must move in at most one dimension.";
    }
    // Ensure all non zero elements are a power of 2. Combined with the
    // broadcast check above this prevents per element swizzling. The intent of
    // the linear component is to rearrange whole rows or cache-line sized
    // chunks of rows.
    if (!llvm::all_of(dimBases, [&](const auto &basis) {
          return llvm::all_of(
              basis, [](auto v) { return v == 0 || llvm::isPowerOf2_32(v); });
        })) {
      return emitError() << "Each offset basis must be 0 or a power of two.";
    }
  }

  return success();
}

PaddedSharedEncodingAttr PaddedSharedEncodingAttr::get(
    MLIRContext *context, ArrayRef<std::pair<unsigned, unsigned>> intervalPads,
    ArrayRef<unsigned> order, ArrayRef<int64_t> shape,
    CTAEncodingAttr ctaLayout) {
  auto outDimNames = standardOutDimNames(context, shape.size());
  StringAttr kOffset = StringAttr::get(context, "offset");

  // Create identity mapping based on shape and order
  LinearLayout linearComponent =
      identityStandardND(kOffset, SmallVector<unsigned>(shape), order);
  linearComponent = combineCtaCgaWithShape(linearComponent, ctaLayout, shape);

  return get(context, intervalPads, linearComponent);
}

PaddedSharedEncodingAttr PaddedSharedEncodingAttr::get(
    MLIRContext *context, ArrayRef<std::pair<unsigned, unsigned>> intervalPads,
    LinearLayout linearComponent) {
  SmallVector<unsigned> intervals, paddings;
  intervals.reserve(intervalPads.size());
  paddings.reserve(intervalPads.size());
  for (auto [interval, padding] : intervalPads) {
    intervals.push_back(interval);
    paddings.push_back(padding);
  }
  return get(context, intervals, paddings, linearComponent);
}

SmallVector<unsigned>
PaddedSharedEncodingAttr::basesPerDim(StringAttr dimName,
                                      bool skipBroadcast) const {
  const auto &ll = getLinearComponent();
  auto rank = ll.getNumOutDims();
  return basesPerDimImpl(ll.getBases(), dimName, rank, skipBroadcast);
}

int64_t PaddedSharedEncodingAttr::getPaddedSize(ArrayRef<int64_t> shape) const {
  int64_t unpaddedSize = product(shape);
  int64_t paddingSize = 0;
  for (auto [interval, padding] :
       llvm::zip_equal(getIntervals(), getPaddings())) {
    paddingSize += (unpaddedSize >> llvm::Log2_32(interval))
                   << llvm::Log2_32(padding);
    // There is no need for padding after the last element
    if (unpaddedSize % interval == 0)
      paddingSize -= padding;
  }
  return unpaddedSize + paddingSize;
}

SmallVector<unsigned>
PaddedSharedEncodingAttr::orderPerDim(StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) const {
  return orderPerDimImpl(getLinearComponent(), dimName, defaultOrder);
}

SmallVector<unsigned> PaddedSharedEncodingAttr::getOrder() const {
  auto rank = getLinearComponent().getNumOutDims();
  SmallVector<unsigned> order(rank);
  // Choose [rank-1, rank-2, ... 0] as the default order in case
  // there are dims that do not move in the offsets
  std::iota(order.rbegin(), order.rend(), 0);

  return orderPerDim(StringAttr::get(getContext(), "offset"), order);
}

CTAEncodingAttr PaddedSharedEncodingAttr::getCTALayout() const {
  auto splitNum = basesPerDim(StringAttr::get(getContext(), "block"));
  return linearToCTAEncodingAttr(getLinearComponent(), splitNum);
}


//===----------------------------------------------------------------------===//
// Sliced Encoding
//===----------------------------------------------------------------------===//

Attribute SliceEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};
  unsigned dim = mlir::cast<IntegerAttr>(attrs.get("dim")).getInt();
  auto parent = mlir::dyn_cast<DistributedEncodingTrait>(attrs.get("parent"));
  if (!parent) {
    parser.emitError(parser.getNameLoc(),
                     "expected a distributed encoding trait");
    return {};
  }
  return parser.getChecked<SliceEncodingAttr>(parser.getContext(), dim, parent);
}

void SliceEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "dim = " << getDim() << ", "
          << "parent = " << getParent() << "}>";
}

LogicalResult
SliceEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          unsigned dim, DistributedEncodingTrait parent) {
  unsigned rank = ::getCTALayout(parent).getRank();
  if (rank <= 1)
    return emitError() << "parent layout must have at least rank >= 2";
  if (dim >= rank) {
    return emitError() << "slice dim=" << dim
                       << " must be less than the parent rank=" << rank;
  }
  return success();
}

SmallVector<unsigned> SliceEncodingAttr::getRepOrder() const {
  auto parentRepOrder = getParent().getRepOrder();
  return eraseOrder(parentRepOrder, getDim());
}

CTAEncodingAttr SliceEncodingAttr::getCTALayout() const {
  auto layout = ::getCTALayout(getParent()).getLinearLayout();
  layout = removeStandardDim(layout, getDim());
  return CTAEncodingAttr::get(getContext(), layout);
}

template <class T>
SmallVector<T> SliceEncodingAttr::paddedShape(ArrayRef<T> shape) const {
  size_t rank = shape.size();
  unsigned dim = getDim();
  SmallVector<T> retShape(rank + 1);
  for (unsigned d = 0; d < rank + 1; ++d) {
    if (d < dim)
      retShape[d] = shape[d];
    else if (d == dim)
      retShape[d] = 1;
    else
      retShape[d] = shape[d - 1];
  }
  return retShape;
}
template SmallVector<unsigned>
SliceEncodingAttr::paddedShape<unsigned>(ArrayRef<unsigned> shape) const;
template SmallVector<int64_t>
SliceEncodingAttr::paddedShape<int64_t>(ArrayRef<int64_t> shape) const;


//===----------------------------------------------------------------------===//
// MMA encoding
//===----------------------------------------------------------------------===//

bool NvidiaMmaEncodingAttr::isVolta() const { return getVersionMajor() == 1; }

bool NvidiaMmaEncodingAttr::isTuring() const {
  return getVersionMajor() == 2 && getVersionMinor() == 1;
}

bool NvidiaMmaEncodingAttr::isAmpere() const { return getVersionMajor() == 2; }

bool NvidiaMmaEncodingAttr::isHopper() const { return getVersionMajor() == 3; }

SmallVector<unsigned> NvidiaMmaEncodingAttr::getRepOrder() const {
  return getMatrixOrder(getRank(), /*rowMajor*/ true);
}

SmallVector<unsigned>
NvidiaMmaEncodingAttr::getRepOrderForOperand(int opIdx) const {
  return getOrderForDotOperand(opIdx, getRank(), /*kContig*/ true);
}

SmallVector<int64_t>
NvidiaMmaEncodingAttr::getRepForOperand(ArrayRef<int64_t> shape, int bitwidth,
                                        int kWidth, int opIdx) const {
  assert(kWidth >= std::max(32 / bitwidth, 1) &&
         "kWidth must be >= max(32 / bitwidth, 1) for this function to be "
         "well-defined");
  auto rank = shape.size();
  // Broadcast long K
  auto warpsPerCTA = to_vector(getWarpsPerCTA());
  auto kDim = opIdx == 0 ? rank - 1 : rank - 2;
  warpsPerCTA[kDim] = 1;

  SmallVector<int> tileSize;
  if (rank == 3) {
    tileSize.push_back(1);
  }
  // warpSizeK * (warpRepK * VecBitWidth)
  auto tileBitWidthK = (isAmpere() && bitwidth == 64) ? (4 * 256) : (4 * 64);
  if (opIdx == 0) {
    // m x k
    tileSize.push_back(16);
    tileSize.push_back(tileBitWidthK / bitwidth);
  } else {
    // k x n
    // Hopper path never uses the n value, since this method is only invoked
    // for in-RF (dotOpEnc) operands, but WGMMA only supports in A to be in RF
    // so it's fine if the n is incorrect here
    tileSize.push_back(tileBitWidthK / bitwidth);
    tileSize.push_back(8);
  }

  SmallVector<int64_t> numRep;
  // Lezcano: This is odd. Why do we always return a vector of size 3?
  if (rank != 3) {
    numRep.push_back(1);
  }
  for (auto [s, size, warp] : llvm::zip(shape, tileSize, warpsPerCTA)) {
    numRep.push_back(std::max<int64_t>(1, s / (size * warp)));
  }
  return numRep;
}

Attribute NvidiaMmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned versionMajor = 0;
  unsigned versionMinor = 0;
  SmallVector<unsigned> warpsPerCTA;
  SmallVector<unsigned> instrShape;
  Attribute ctaAttr = nullptr;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "versionMajor") {
      if (parseUInt(parser, attr, versionMajor, "versionMajor").failed())
        return {};
    }
    if (attr.getName() == "versionMinor") {
      if (parseUInt(parser, attr, versionMinor, "versionMinor").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "CGALayout") {
      ctaAttr = attr.getValue();
      continue;
    }
    if (attr.getName() == "instrShape") {
      if (parseIntArrayAttr(parser, attr, instrShape, "instrShape").failed()) {
        return {};
      }
    }
  }

  std::optional<CTAEncodingAttr> CTALayout =
      parseCTAAttr(parser, ctaAttr, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<NvidiaMmaEncodingAttr>(
      parser.getContext(), versionMajor, versionMinor, warpsPerCTA, *CTALayout,
      instrShape);
}

void NvidiaMmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "versionMajor = " << getVersionMajor()
          << ", versionMinor = " << getVersionMinor() //
          << ", warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]";

  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getRank());

  printer << ", instrShape = [" << getInstrShape() << "]}>";
}

//===----------------------------------------------------------------------===//
// NVMMAShared encoding
//===----------------------------------------------------------------------===//

Attribute NVMMASharedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned swizzlingByteWidth;
  bool transposed = false;
  bool fp4Padded = false;
  unsigned elementBitWidth;
  unsigned layoutRank = 2;
  Attribute ctaAttr = nullptr;
  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "swizzlingByteWidth") {
      if (parseUInt(parser, attr, swizzlingByteWidth, "swizzlingByteWidth")
              .failed())
        return {};
    } else if (attr.getName() == "transposed") {
      if (parseBool(parser, attr, transposed, "transposed").failed())
        return {};
    } else if (attr.getName() == "elementBitWidth") {
      if (parseUInt(parser, attr, elementBitWidth, "elementBitWidth").failed())
        return {};
    } else if (attr.getName() == "fp4Padded") {
      if (parseBool(parser, attr, fp4Padded, "fp4Padded").failed())
        return {};
    } else if (attr.getName() == "CGALayout") {
      ctaAttr = attr.getValue();
    } else if (attr.getName() == "rank") {
      if (parseUInt(parser, attr, layoutRank, "rank").failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  std::optional<CTAEncodingAttr> CTALayout =
      parseCTAAttr(parser, ctaAttr, layoutRank);
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<NVMMASharedEncodingAttr>(
      parser.getContext(), swizzlingByteWidth, transposed, elementBitWidth,
      fp4Padded, *CTALayout);
}

void NVMMASharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "swizzlingByteWidth = " << getSwizzlingByteWidth() //
          << ", transposed = " << getTransposed()               //
          << ", elementBitWidth = " << getElementBitWidth();
  if (getFp4Padded()) {
    // Print only in this case to reduce the noise for the more common case.
    printer << ", fp4Padded = true";
  }
  unsigned rank = getCTALayout().getCTAOrder().size();
  auto *ctx = getContext();
  auto defaultLayout = CTAEncodingAttr::getDefault(ctx, rank);
  if (getCTALayout() == defaultLayout && rank != 2) {
    printer << ", rank = " << rank;
  } else {
    maybePrintCTALayout(ctx, printer, getCTALayout(), rank);
  }
  printer << "}>";
}

int NVMMASharedEncodingAttr::getVec() const {
  if (getSwizzlingByteWidth() == 0)
    return 1;
  return 128 / getElementBitWidth();
}

int NVMMASharedEncodingAttr::getPerPhase() const {
  if (getSwizzlingByteWidth() == 0)
    return 1;
  return 128 / getSwizzlingByteWidth();
}

int NVMMASharedEncodingAttr::getMaxPhase() const {
  if (getSwizzlingByteWidth() == 0)
    return 1;
  return getSwizzlingByteWidth() / 16;
}

int32_t NVMMASharedEncodingAttr::getAlignment() const {
  return 128 * getMaxPhase();
}