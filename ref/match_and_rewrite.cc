#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

struct FuseGatherV2OpPattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;
public:
  LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const override {
    llvm::outs() << "BEGIN:\n";
    llvm::outs() << "================================================================================================================" << "\n";
    llvm::outs() << *funcOp << "\n";
    llvm::outs() << "================================================================================================================" << "\n";
    llvm::outs() << "\n";

    if (!isValid(funcOp, kPatternName)) {
      return failure();
    }

    OpBuilder builder(funcOp);
    TF::GatherV2Op firstGatherOp(nullptr);
    auto returnOp = cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());
    auto returnOperands = returnOp.getOperands();

    SmallVector<Value, 4> paramsMems;
    SmallVector<Value, 4> indicesMems;
    Value axis;
    unsigned i = 0;
    for (auto& op : funcOp.getBody().front().getOperations()) {
      if (auto curGatherOp = dyn_cast<TF::GatherV2Op>(op)) {
        auto operands = curGatherOp.getOperands();
        auto params = operands[0].getDefiningOp()->getOperand(0);
        auto indices = operands[1].getDefiningOp()->getOperand(0);
        paramsMems.push_back(params);
        indicesMems.push_back(indices);
        if (!firstGatherOp) {
          firstGatherOp = curGatherOp;
          axis = operands[2].getDefiningOp()->getOperand(0);
        }
        i++;
      }
    }

    if (!firstGatherOp) {
      return failure();
    }

    //=====================================================================================
    // Start transformation
    //=====================================================================================
    rewriter.setInsertionPoint(returnOp);

    // Constants.
    Value c0 = rewriter.create<arith::ConstantIndexOp>(firstGatherOp.getLoc(), 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(firstGatherOp.getLoc(), 1);
    Value numPairs = rewriter.create<arith::ConstantIndexOp>(firstGatherOp.getLoc(), paramsMems.size());

    // Extract dynamic dimensions (all params and indices have the same shape).
    Value paramsDim0 = rewriter.create<memref::DimOp>(firstGatherOp.getLoc(), paramsMems[0], c0);
    Value paramsDim1 = rewriter.create<memref::DimOp>(firstGatherOp.getLoc(), paramsMems[0], c1);
    Value indicesDim0 = rewriter.create<memref::DimOp>(firstGatherOp.getLoc(), indicesMems[0], c0);
    Value axisVal = rewriter.create<memref::LoadOp>(firstGatherOp.getLoc(), axis, ValueRange{});

    // Allocate GPU output buffers.
    // auto gpuGlobal = rewriter.getI32IntegerAttr(0);
    // SmallVector<Value, 4> outputMems;
    // SmallVector<Type, 4> outputTypes;
    // auto outMemType = MemRefType::get({-1, -1}, rewriter.getF32Type(), MemRefLayoutAttrInterface{}, gpuGlobal);
    // for (int i = 0; i < paramsMems.size(); ++i) {
    //   Location loc = NameLoc::get(StringAttr::get(getContext(), "memref::AllocOp " + std::to_string(i)));
    //   outputMems.push_back(rewriter.create<memref::AllocOp>(loc, outMemType, ValueRange{indicesDim0, paramsDim1}));
    //   outputTypes.push_back(outMemType);

    //   Location loadOpLoc = NameLoc::get(StringAttr::get(getContext(), "memref::LoadOp paramsMems[" + std::to_string(i) + "]"));
    //   Value v = rewriter.create<memref::LoadOp>(loadOpLoc, paramsMems[i], ValueRange{c0, c1});
    //   Location storeOpLoc = NameLoc::get(StringAttr::get(getContext(), "memref::StoreOp outputMems[" + std::to_string(i) + "]"));
    //   rewriter.create<memref::StoreOp>(storeOpLoc, v, outputMems[i], ValueRange{c0, c1});
    // }

    // Single GPU kernel for all pairs.
    Location launchOpLoc = NameLoc::get(StringAttr::get(getContext(), "gpu::LaunchOp"));
    auto launchOp = rewriter.create<gpu::LaunchOp>(
      launchOpLoc, numPairs, c1, c1, // Grid size: num_pairs x 1 x 1
      indicesDim0, paramsDim1, c1 // Block size: indices_size x params_dim1 x 1
    );
    rewriter.setInsertionPointToStart(&launchOp.getBody().front());

    Value pairIdx = rewriter.create<gpu::BlockIdOp>(launchOpLoc, rewriter.getIndexType(), gpu::Dimension::x);
    Value tx = rewriter.create<gpu::ThreadIdOp>(launchOpLoc, rewriter.getIndexType(), gpu::Dimension::x);
    Value ty = rewriter.create<gpu::ThreadIdOp>(launchOpLoc, rewriter.getIndexType(), gpu::Dimension::y);

    // Select params, indices, and output based on pair_idx.
    Value paramsMem = paramsMems[0];
    Value indicesMem = indicesMems[0];
    // Value outMem = outputMems[0];
    for (int i = 1; i < paramsMems.size(); ++i) {
      Value pairConst = rewriter.create<arith::ConstantIndexOp>(launchOpLoc, i);
      Value cond = rewriter.create<arith::CmpIOp>(launchOpLoc, arith::CmpIPredicate::eq, pairIdx, pairConst);
      Location paramsMemLoc = NameLoc::get(StringAttr::get(getContext(), "arith::SelectOp paramsMem"));
      paramsMem = rewriter.create<arith::SelectOp>(paramsMemLoc, cond, paramsMems[i], paramsMem);
      Location indicesMemLoc = NameLoc::get(StringAttr::get(getContext(), "arith::SelectOp indicesMem"));
      indicesMem = rewriter.create<arith::SelectOp>(indicesMemLoc, cond, indicesMems[i], indicesMem);
      // Location outMemLoc = NameLoc::get(StringAttr::get(getContext(), "arith::SelectOp outMem"));
      // outMem = rewriter.create<arith::SelectOp>(outMemLoc, cond, outputMems[i], outMem);
    }

    // Gather logic.
    Location loadOpLoc = NameLoc::get(StringAttr::get(getContext(), "memref::LoadOp indicesMem"));
    Value idxi32 = rewriter.create<memref::LoadOp>(loadOpLoc, indicesMem, ValueRange{tx});
    Value idx = rewriter.create<arith::IndexCastOp>(loadOpLoc, rewriter.getIndexType(), idxi32);
    Value zeroi32 = rewriter.create<arith::ConstantOp>(launchOpLoc, builder.getIntegerAttr(builder.getI32Type(), 0));
    Value cond = rewriter.create<arith::CmpIOp>(launchOpLoc, arith::CmpIPredicate::sgt, axisVal, zeroi32);
    auto ifOp = rewriter.create<scf::IfOp>(launchOpLoc, rewriter.getF32Type(), cond, /*hasElse=*/true);
    {
      // then block
      Block* thenBlock = ifOp.thenBlock();
      rewriter.setInsertionPointToStart(thenBlock);
      Location thenLoc = NameLoc::get(StringAttr::get(getContext(), "memref::LoadOp paramsMem in thenBlock"));
      Value thenVal = rewriter.create<memref::LoadOp>(thenLoc, paramsMem, ValueRange{idx, ty});
      rewriter.create<scf::YieldOp>(launchOpLoc, thenVal);
    }
    {
      // else block
      Block* elseBlock = ifOp.elseBlock();
      rewriter.setInsertionPointToStart(elseBlock);
      Location elseLoc = NameLoc::get(StringAttr::get(getContext(), "memref::LoadOp paramsMem in elseBlock"));
      Value elseVal = rewriter.create<memref::LoadOp>(elseLoc, paramsMem, ValueRange{ty, idx});
      rewriter.create<scf::YieldOp>(launchOpLoc, elseVal);
    }

    rewriter.setInsertionPointAfter(ifOp);
    // Location storeOpLoc = NameLoc::get(StringAttr::get(getContext(), "memref::StoreOp outMem"));
    // rewriter.create<memref::StoreOp>(storeOpLoc, ifOp.getResult(0), outMem, ValueRange{tx, ty});
    // Create gpu.printf operation
    rewriter.create<gpu::PrintfOp>(
      UnknownLoc::get(getContext()),
      rewriter.getStringAttr("Thread %d\n"),  // Format string
      ValueRange{tx});                        // Arguments
    rewriter.create<gpu::TerminatorOp>(launchOpLoc);

    rewriter.setInsertionPointAfter(launchOp);

    // auto gpuGlobal = rewriter.getI32IntegerAttr(0);
    SmallVector<Value, 4> outputMems;
    SmallVector<Type, 4> outputTypes;
    Value d0 = rewriter.create<arith::ConstantIndexOp>(firstGatherOp.getLoc(), 2);
    Value d1 = rewriter.create<arith::ConstantIndexOp>(firstGatherOp.getLoc(), 4);
    auto outMemType = MemRefType::get({2, 4}, rewriter.getF32Type(), MemRefLayoutAttrInterface{});
    for (int i = 0; i < paramsMems.size(); ++i) {
      Location loc = NameLoc::get(StringAttr::get(getContext(), "memref::AllocOp " + std::to_string(i)));
      outputMems.push_back(rewriter.create<memref::AllocOp>(loc, outMemType));
      outputTypes.push_back(outMemType);

      Location loadOpLoc = NameLoc::get(StringAttr::get(getContext(), "memref::LoadOp paramsMems[" + std::to_string(i) + "]"));
      Value v = rewriter.create<memref::LoadOp>(loadOpLoc, paramsMems[i], ValueRange{c0, c1});
      Location storeOpLoc = NameLoc::get(StringAttr::get(getContext(), "memref::StoreOp outputMems[" + std::to_string(i) + "]"));
      rewriter.create<memref::StoreOp>(storeOpLoc, v, outputMems[i], ValueRange{c0, c1});
    }

    FunctionType funcType = FunctionType::get(getContext(), funcOp.getFunctionType().getInputs(), outputTypes);
    funcOp.setFunctionType(funcType);
    returnOp.operandsMutable().assign(outputMems);
    funcOp->setAttr(AF_BATCH_COMPUTE_PATTERN, builder.getStringAttr(std::string(kPatternName).append("Done").c_str()));

    llvm::outs() << "END:\n";
    llvm::outs() << "================================================================================================================" << "\n";
    llvm::outs() << *funcOp << "\n";
    llvm::outs() << "================================================================================================================" << "\n";
    llvm::outs() << "\n";

    return success(); 
  }
};

//================================================================================
// Create a TF::ConstOp from axisVal
//================================================================================
// RankedTensorType tensorType = RankedTensorType::get({}, builder.getI32Type());
// Attribute scalarAttr = builder.getI32IntegerAttr(axisVal);
// DenseElementsAttr valueAttr = DenseElementsAttr::get(tensorType, scalarAttr);
// Operation* newAxisOp = builder.create<TF::ConstOp>(builder.getUnknownLoc(), tensorType, valueAttr);
// Value newAxisValue = newAxisOp->getResult(0);

//================================================================================
// Extract int32
//================================================================================
// int32_t axisVal = 0;
// auto axisOp = dyn_cast<tf_executor::IslandOp>(axis.getDefiningOp());
// if (auto tfAxisOp = islandOpCast<TF::ConstOp>(axisOp)) {
//   auto axisAttr = tfAxisOp.value().cast<DenseIntElementsAttr>();
//   axisVal = axisAttr.getValues<int32_t>()[0];
//   llvm::outs() << "axisVal: " << axisVal << "\n";
// } else {
//   ignoreGatherOps.insert(ignoreGatherOps.end(), cluster.begin(), cluster.end());
//   cluster.clear();
//   return failure();
// }

//================================================================================
// Shape and Type
//================================================================================
// ArrayRef<int64_t> paramsShape;
// ::mlir::Type paramsType;
// ArrayRef<int64_t> indicesShape;
// ::mlir::Type indicesType;
// paramsShape = gatherOp.params().getType().cast<TensorType>().getShape();
// paramsType = gatherOp.Tparams();
// indicesShape = gatherOp.indices().getType().cast<TensorType>().getShape();
// indicesType = gatherOp.Tindices();
// axis = gatherOp.axis();
// axisSet.insert(gatherOp.axis().getDefiningOp());

SmallVector<std::vector<Operation *>> Cluster::split() {
  const int kClusterSizeThreshold = auto_fusion::AFClusterSizeThreshold();
  SmallVector<std::vector<Operation *>, 2> splitClusters;
  if (cluster.size() <= kClusterSizeThreshold) {
    splitClusters.push_back(cluster);
  } else {
    int parts = cluster.size() / kClusterSizeThreshold;
    parts += (cluster.size() % kClusterSizeThreshold == 0) ? 0 : 1;
    for (int i = 0; i < parts - 1; i++) {
      std::vector<Operation *> partCluster(cluster.begin() + (i * kClusterSizeThreshold), cluster.begin() + (i + 1) * kClusterSizeThreshold);
      splitClusters.push_back(std::move(partCluster));
    }

    std::vector<Operation *> partCluster(cluster.begin() + (parts - 1) * kClusterSizeThreshold, cluster.end());
    splitClusters.push_back(std::move(partCluster));
  }

  return splitClusters;
}