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


  //=============================================================================
  // matchAndRewrite_v1
  //=============================================================================
  static void matchAndRewrite_v1(func::FuncOp funcOp) {
    if (!isValid(funcOp, std::string(kPatternName))) {
      return;
    }

    auto& sonyOs = getSonyOs("FuseSumOpPattern.mlir");

    {
      std::lock_guard<std::mutex> lk(sony_mutex);
      sonyOs << "// Before matchAndRewrite\n";
      sonyOs << funcOp << "\n";
      sonyOs << "// ======================================================================================\n\n";
      sonyOs.flush();
    }

    // Rewrite entire body
    OpBuilder rewriter(funcOp);
    int argSize = funcOp.getNumArguments();
    auto context = funcOp.getContext();
    auto loc = funcOp.getLoc();
    auto oldEntryBlock = &funcOp.getBody().front();

    Block* newEntryBlock = new Block();
    funcOp.getBody().push_front(newEntryBlock);
    for (auto& arg : oldEntryBlock->getArguments()) {
      newEntryBlock->addArgument(arg.getType(), arg.getLoc());
    }

    oldEntryBlock->erase();

    rewriter.setInsertionPointToStart(newEntryBlock);

    // Create constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c0_i32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    MemRefType inputType = funcOp.getArgument(0).getType().cast<mlir::MemRefType>();
    int n = inputType.getRank();
    int k = 0;
    bool keepDims = false;
    int counter = 0;
    auto reductionIndexAttr = funcOp->getAttr(tfext_kernel_gen::getReductionIndicesAttrKey());
    if (reductionIndexAttr) {
      k = reductionIndexAttr.cast<IntegerAttr>().getInt();
      counter++;
    }

    auto keepDimsAttr = funcOp->getAttr(tfext_kernel_gen::getKeepDimsAttrKey());
    if (keepDimsAttr) {
      keepDims = keepDimsAttr.cast<BoolAttr>().getValue();
      counter++;
    }

    assert(counter == 2 && "Failed to get either reduction_indices or keep_dims");
    assert(k >= 0 && k < n && "k must be within inputType rank");

    SmallVector<Value>  dims;
    SmallVector<int64_t>  outputDims;

    // Build dims
    int idx = 0;
    SmallVector<Value> dynamicDims;
    for (auto dim : inputType.getShape()) {
      Value dimVal;
      if (dim == mlir::ShapedType::kDynamicSize) {
        dimVal = rewriter.create<memref::DimOp>(loc, funcOp.getArgument(0), idx).getResult();
        dims.push_back(dimVal);
      } else {
        dimVal = rewriter.create<arith::ConstantIndexOp>(loc, dim).getResult();
        dims.push_back(dimVal);
      }

      if (idx != k) {
        outputDims.push_back(dim);

        if (dim == mlir::ShapedType::kDynamicSize) {
          dynamicDims.push_back(dimVal);
        }
      }

      idx++;
    }

    std::vector<Value> outputs;
    std::vector<Value> initVals;
    for (int i = 0; i < argSize; i++) {
      auto outputType = MemRefType::get(ArrayRef<int64_t>(outputDims), inputType.getElementType());
      auto outputTensor = rewriter.create<memref::AllocaOp>(loc, outputType, ArrayRef<Value>(dynamicDims)).getResult();
      outputs.push_back(outputTensor);
      initVals.push_back(c0_i32);
    }

    // Reduction logic
    {
      OpBuilder::InsertionGuard guard(rewriter);

      // Create nested scf.parallel loops for all dimensions except k
      SmallVector<scf::ParallelOp> parallelOps;
      SmallVector<Value> inIndices;  // Induction vars for non-k dimensions
      SmallVector<Value> outIndices;  // n-1 indices

      for (int i = 0; i < n; i++) {
        if (i == k) {
          inIndices.push_back(Value());
          continue;
        }

        auto par = rewriter.create<scf::ParallelOp>(loc, ValueRange{c0}, ValueRange{dims[i]}, ValueRange{c1});
        parallelOps.push_back(par);
        auto iv0 = par.getInductionVars()[0];
        inIndices.push_back(iv0);
        outIndices.push_back(iv0);
        rewriter.setInsertionPointToStart(par.getBody());
      }

      // Initialize outputTensor to 0
      for (int i = 0; i < argSize; i++) {
        rewriter.create<memref::StoreOp>(loc, c0_i32, outputs[i], outIndices);
      }

      auto forOp = rewriter.create<scf::ForOp>(loc, c0, dims[k], c1, initVals);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forOp.getBody());
        inIndices[k] = forOp.getInductionVar();

        std::vector<Value> accResults;
        for (int i = 0; i < argSize; i++) {
          Value init = forOp.getBody()->getArgument(i+1);
          Value loadVal = rewriter.create<memref::LoadOp>(loc, funcOp.getArgument(i), inIndices);
          Value addVal = rewriter.create<arith::AddIOp>(loc, init, loadVal);
          accResults.push_back(addVal);
        }

        rewriter.create<scf::YieldOp>(loc, accResults);
      }

      // Store the result
      for (int i = 0; i < argSize; i++) {
        rewriter.create<memref::StoreOp>(loc, forOp.getResult(i), outputs[i], outIndices);
      }
    }

    if (keepDims) {
      auto resultTypes = funcOp.getResultTypes();
      MemRefType shapeType = MemRefType::get({n}, rewriter.getIndexType());
      auto shapeMemRef = rewriter.create<mlir::memref::AllocaOp>(loc, shapeType);

      // Fill shape memref with constants
      for (int i = 0; i < n; i++) {
        Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
        if (i != k) {
          rewriter.create<memref::StoreOp>(loc, dims[i], shapeMemRef, idx);
        } else {
          // dim = 1
          rewriter.create<memref::StoreOp>(loc, c1, shapeMemRef, idx);
        }
      }

      for (int i = 0; i < argSize; i++) {
        auto reshapedOutputTensor = rewriter.create<memref::ReshapeOp>(loc, resultTypes[i+1], outputs[i], shapeMemRef.getResult());
        outputs[i] = reshapedOutputTensor.getResult();
      }
    }

    rewriter.setInsertionPointToEnd(newEntryBlock);
    rewriter.create<func::ReturnOp>(loc, outputs);

    // {
    //   std::lock_guard<std::mutex> lk(sony_mutex);
    //   sonyOs << "// After matchAndRewrite\n";
    //   sonyOs << funcOp << "\n";
    //   sonyOs.flush();
    // }

    // Explicitly verify after rewrite
    if (failed(mlir::verify(funcOp))) {
      funcOp.emitError("funcOp is invalid after rewrite");
    }
  }


  static LogicalResult matchAndRewrite_v2(func::FuncOp funcOp) {
    if (!isValid(funcOp, std::string(kPatternName))) {
      return failure();
    }

    if (!funcOp.getArgument(0).getType().isa<tf::OpKernelContextType>()) {
      return failure();
    }

    auto& sonyOs = getSonyOs("FuseSumOpPattern.mlir");

    {
      std::lock_guard<std::mutex> lk(sony_mutex);
      sonyOs << "// Before matchAndRewrite\n";
      funcOp->print(sonyOs, OpPrintingFlags().printGenericOpForm());
      sonyOs << "\n";
      sonyOs << "// ======================================================================================\n\n";
      sonyOs.flush();
    }

    // Rewrite entire body
    // Register the dialect (e.g., ArithDialect) as legal
    auto context = funcOp.getContext();
    auto gpuDialect = context->getOrLoadDialect<gpu::GPUDialect>();
    if (!gpuDialect) {
      llvm::outs() << "error: failed to load gpu::GPUDialect\n";
      std::terminate();
    }

    OpBuilder rewriter(funcOp);
    int argSize = funcOp.getNumArguments();
    auto loc = funcOp.getLoc();
    auto oldEntryBlock = &funcOp.getBody().front();

    Block* newEntryBlock = new Block();
    funcOp.getBody().push_front(newEntryBlock);
    for (auto& arg : oldEntryBlock->getArguments()) {
      newEntryBlock->addArgument(arg.getType(), arg.getLoc());
    }

    oldEntryBlock->erase();

    Value tfOpCtx = funcOp.getArgument(0);
    rewriter.setInsertionPointToStart(newEntryBlock);

    // Create constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c0_i32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    MemRefType inputType = funcOp.getArgument(1).getType().cast<mlir::MemRefType>();
    int n = inputType.getRank();
    int k = 0;
    bool keepDims = false;
    int counter = 0;
    auto reductionIndexAttr = funcOp->getAttr(tfext_kernel_gen::getReductionIndicesAttrKey());
    if (reductionIndexAttr) {
      k = reductionIndexAttr.cast<IntegerAttr>().getInt();
      counter++;
    }

    auto keepDimsAttr = funcOp->getAttr(tfext_kernel_gen::getKeepDimsAttrKey());
    if (keepDimsAttr) {
      keepDims = keepDimsAttr.cast<BoolAttr>().getValue();
      counter++;
    }

    assert(counter == 2 && "Failed to get either reduction_indices or keep_dims");
    assert(k >= 0 && k < n && "k must be within inputType rank");

    SmallVector<Value>  dims;
    SmallVector<int64_t>  outputDims;

    // Build dims
    int idx = 0;
    SmallVector<Value> dynamicDims;
    for (auto dim : inputType.getShape()) {
      Value dimVal;
      if (dim == mlir::ShapedType::kDynamicSize) {
        dimVal = rewriter.create<memref::DimOp>(loc, funcOp.getArgument(1), idx).getResult();
        dims.push_back(dimVal);
      } else {
        dimVal = rewriter.create<arith::ConstantIndexOp>(loc, dim).getResult();
        dims.push_back(dimVal);
      }

      if (idx != k) {
        outputDims.push_back(dim);

        if (dim == mlir::ShapedType::kDynamicSize) {
          dynamicDims.push_back(dimVal);
        }
      }

      idx++;
    }

    std::vector<Value> outputs;
    std::vector<Value> initVals;
    for (int i = 1; i < argSize; i++) {
      auto outputType = MemRefType::get(ArrayRef<int64_t>(outputDims), inputType.getElementType());
      // auto outputTensor = rewriter.create<memref::AllocaOp>(loc, outputType, ArrayRef<Value>(dynamicDims)).getResult();
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memref_type, Value ctx, ValueRange dyn_sizes);
      Value outputTensor = rewriter.create<tf::TFAllocOp>(loc, outputType, tfOpCtx, dynamicDims);
      Value cond = rewriter.create<tf::IsValidMemRefOp>(loc, rewriter.getIntegerType(1), outputTensor);
      rewriter.create<tf::TFAssertOp>(loc, tfOpCtx, cond, tf::ErrorCode::RESOURCE_EXHAUSTED, "failed to allocate memory");

      outputs.push_back(outputTensor);
      initVals.push_back(c0_i32);
    }

    //========================================================================================
    // Create GPU kernel
    //========================================================================================
    Value c80 = rewriter.create<arith::ConstantIndexOp>(loc, 80);
    Value c512 = rewriter.create<arith::ConstantIndexOp>(loc, 512);

    // GPU blocks
    Value blockCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, dims[0], c80);
    Value numBlocks = rewriter.create<arith::SelectOp>(loc, blockCond, dims[0], c80);

    // GPU threads
    Value threadCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, dims[1], c512);
    Value numThreads = rewriter.create<arith::SelectOp>(loc, threadCond, dims[1], c512);

    auto launchOp = rewriter.create<gpu::LaunchOp>(
      loc, numBlocks, c1, c1, // Grid size: numBlocks x 1 x 1
      numThreads, c1, c1 // Block size: numThreads x 1 x 1
    );

    // Reduction logic
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&launchOp.getBody().front());
      Value numBlocks = launchOp.getGridSizeX();
      Value blockIdx = rewriter.create<gpu::BlockIdOp>(loc, rewriter.getIndexType(), gpu::Dimension::x);
      Value estBlockStep = rewriter.create<arith::DivUIOp>(loc, dims[0], numBlocks);
      Value blockCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, dims[0], numBlocks);
      Value blockStep = rewriter.create<arith::SelectOp>(loc, blockCond, c1, estBlockStep);
      Value blockLb = rewriter.create<arith::MulIOp>(loc, blockIdx, blockStep);
      Value estBlockUb = rewriter.create<arith::AddIOp>(loc, blockLb, blockStep);
      Value remBlocks = rewriter.create<arith::RemUIOp>(loc, dims[0], numBlocks);
      Value maxBlockUb = rewriter.create<arith::AddIOp>(loc, estBlockUb, remBlocks);
      Value blockIdxDelta = rewriter.create<arith::SubIOp>(loc, numBlocks, blockIdx);
      Value blockUbCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, blockIdxDelta, c1);
      Value blockUb = rewriter.create<arith::SelectOp>(loc, blockUbCond, estBlockUb, maxBlockUb);

      Value numThreads = launchOp.getBlockSizeX();
      Value threadIdx = rewriter.create<gpu::ThreadIdOp>(loc, rewriter.getIndexType(), gpu::Dimension::x);
      Value estThreadStep = rewriter.create<arith::DivUIOp>(loc, dims[1], numThreads);
      Value threadCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, dims[1], numThreads);
      Value threadStep = rewriter.create<arith::SelectOp>(loc, threadCond, c1, estThreadStep);
      Value threadLb = rewriter.create<arith::MulIOp>(loc, threadIdx, threadStep);
      Value estThreadUb = rewriter.create<arith::AddIOp>(loc, threadLb, threadStep);
      Value remThreads = rewriter.create<arith::RemUIOp>(loc, dims[1], numThreads);
      Value maxThreadUb = rewriter.create<arith::AddIOp>(loc, estThreadUb, remThreads);
      Value threadIdxDelta = rewriter.create<arith::SubIOp>(loc, numThreads, threadIdx);
      Value threadUbCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, threadIdxDelta, c1);
      Value threadUb = rewriter.create<arith::SelectOp>(loc, threadUbCond, estThreadUb, maxThreadUb);

      SmallVector<Value> inIndices;  // Induction vars for non-k dimensions
      SmallVector<Value> outIndices;  // n-1 indices

      // Block level loop
      auto blockForOp = rewriter.create<scf::ForOp>(loc, blockLb, blockUb, c1);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(blockForOp.getBody());
        inIndices.push_back(blockForOp.getInductionVar());
        outIndices.push_back(blockForOp.getInductionVar());

        auto threadForOp = rewriter.create<scf::ForOp>(loc, threadLb, threadUb, c1);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(threadForOp.getBody());
          inIndices.push_back(threadForOp.getInductionVar());
          outIndices.push_back(threadForOp.getInductionVar());

          // Initialize outputTensor to 0
          for (int i = 1; i < argSize; i++) {
            rewriter.create<memref::StoreOp>(loc, c0_i32, outputs[i], outIndices);
          }

          // Reduction
          auto reduceForOp = rewriter.create<scf::ForOp>(loc, c0, dims[k], c1, initVals);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(reduceForOp.getBody());
            inIndices.push_back(reduceForOp.getInductionVar());

            std::vector<Value> accResults;
            for (int i = 1; i < argSize; i++) {
              Value init = reduceForOp.getBody()->getArgument(i);
              Value loadVal = rewriter.create<memref::LoadOp>(loc, funcOp.getArgument(i), inIndices);
              Value addVal = rewriter.create<arith::AddIOp>(loc, init, loadVal);
              accResults.push_back(addVal);
            }

            rewriter.create<scf::YieldOp>(loc, accResults);
          }

          // Store the result
          for (int i = 1; i < argSize; i++) {
            rewriter.create<memref::StoreOp>(loc, reduceForOp.getResult(i), outputs[i], outIndices);
          }
        }
      }

      rewriter.create<gpu::TerminatorOp>(loc);
    }
    //========================================================================================

    if (keepDims) {
      auto resultTypes = funcOp.getResultTypes();
      MemRefType shapeType = MemRefType::get({n}, rewriter.getIndexType());
      auto shapeMemRef = rewriter.create<mlir::memref::AllocaOp>(loc, shapeType);

      // Fill shape memref with constants
      for (int i = 0; i < n; i++) {
        Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
        if (i != k) {
          rewriter.create<memref::StoreOp>(loc, dims[i], shapeMemRef, idx);
        } else {
          // dim = 1
          rewriter.create<memref::StoreOp>(loc, c1, shapeMemRef, idx);
        }
      }

      for (int i = 1; i < argSize; i++) {
        auto reshapedOutputTensor = rewriter.create<memref::ReshapeOp>(loc, resultTypes[i], outputs[i], shapeMemRef.getResult());
        outputs[i] = reshapedOutputTensor.getResult();
      }
    }

    rewriter.setInsertionPointToEnd(newEntryBlock);
    rewriter.create<func::ReturnOp>(loc, outputs);

    {
      std::lock_guard<std::mutex> lk(sony_mutex);
      sonyOs << "// After matchAndRewrite\n";
      funcOp->print(sonyOs, OpPrintingFlags().printGenericOpForm());
      sonyOs << "\n";
      sonyOs.flush();
    }

    // Explicitly verify after rewrite
    // if (failed(mlir::verify(funcOp))) {
    //   funcOp.emitError("funcOp is invalid after rewrite");
    // }

    return success();
  }