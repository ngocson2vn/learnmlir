# MLIR Legalization
Call stack:
### 1. Loop over operations
llvm-project/mlir/lib/Transforms/Utils/DialectConversion.cpp:
```C++
LogicalResult OperationConverter::convertOperations(ArrayRef<Operation *> ops) {
  for (auto *op : toConvert) {
    if (failed(convert(op))) {
      // Dialect conversion failed.
      if (rewriterImpl.config.allowPatternRollback) {
        // Rollback is allowed: restore the original IR.
        rewriterImpl.undoRewrites();
      } else {
        // Rollback is not allowed: apply all modifications that have been
        // performed so far.
        rewriterImpl.applyRewrites();
      }
      return failure();
    }
  }
}
```

### 2. Convert op
llvm-project/mlir/lib/Transforms/Utils/DialectConversion.cpp:
```C++
LogicalResult OperationConverter::convert(Operation *op) {
  const ConversionConfig &config = rewriter.getConfig();

  // Legalize the given operation.
  if (failed(opLegalizer.legalize(op))) {
    // Handle the case of a failed conversion for each of the different modes.
    // Full conversions expect all operations to be converted.
    if (mode == OpConversionMode::Full)
      return op->emitError()
             << "failed to legalize operation '" << op->getName() << "'";
    // Partial conversions allow conversions to fail iff the operation was not
    // explicitly marked as illegal. If the user provided a `unlegalizedOps`
    // set, non-legalizable ops are added to that set.
    if (mode == OpConversionMode::Partial) {
      if (opLegalizer.isIllegal(op))
        return op->emitError()
               << "failed to legalize operation '" << op->getName()
               << "' that was explicitly marked illegal";
      if (config.unlegalizedOps)
        config.unlegalizedOps->insert(op);
    }
  } else if (mode == OpConversionMode::Analysis) {
    // Analysis conversions don't fail if any operations fail to legalize,
    // they are only interested in the operations that were successfully
    // legalized.
    if (config.legalizableOps)
      config.legalizableOps->insert(op);
  }
  return success();
}
```

### 3. Apply user-defined conversion pattern
llvm-project/mlir/lib/Transforms/Utils/DialectConversion.cpp
```C++
LogicalResult OperationLegalizer::legalizeWithPattern(Operation *op) {
  // ...
}
```

### 4. Execute `onSuccess` lambda function
llvm-project/mlir/lib/Rewrite/PatternApplicator.cpp
```C++
LogicalResult PatternApplicator::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter,
    function_ref<bool(const Pattern &)> canApply,
    function_ref<void(const Pattern &)> onFailure,
    function_ref<LogicalResult(const Pattern &)> onSuccess) {
  // ...
          // Process the result of the pattern application.
          if (succeeded(result) && onSuccess && failed(onSuccess(*bestPattern)))
            result = failure();
  // ...
}
```

### 5. Step into `onSuccess`
llvm-project/mlir/lib/Transforms/Utils/DialectConversion.cpp
```C++
  // Functor that performs additional legalization when a pattern is
  // successfully applied.
  auto onSuccess = [&](const Pattern &pattern) {
    assert(rewriterImpl.pendingRootUpdates.empty() && "dangling root updates");
    if (!rewriterImpl.config.allowPatternRollback) {
      // Eagerly erase unused materializations.
      for (auto op : rewriterImpl.patternMaterializations) {
        if (op->use_empty()) {
          rewriterImpl.unresolvedMaterializations.erase(op);
          op.erase();
        }
      }
      rewriterImpl.patternMaterializations.clear();
    }
    SetVector<Operation *> newOps = moveAndReset(rewriterImpl.patternNewOps);
    SetVector<Operation *> modifiedOps =
        moveAndReset(rewriterImpl.patternModifiedOps);
    SetVector<Block *> insertedBlocks =
        moveAndReset(rewriterImpl.patternInsertedBlocks);
    auto result = legalizePatternResult(op, pattern, curState, newOps,
                                        modifiedOps, insertedBlocks);
    appliedPatterns.erase(&pattern);
    if (failed(result)) {
      if (!rewriterImpl.config.allowPatternRollback)
        reportNewIrLegalizationFatalError(pattern, newOps, modifiedOps,
                                          insertedBlocks);
      rewriterImpl.resetState(curState, pattern.getDebugName());
    }
    if (config.listener)
      config.listener->notifyPatternEnd(pattern, result);
    return result;
  };
```

### 6. legalizePatternResult
llvm-project/mlir/lib/Transforms/Utils/DialectConversion.cpp
```C++
LogicalResult OperationLegalizer::legalizePatternResult(
    Operation *op, const Pattern &pattern, const RewriterState &curState,
    const SetVector<Operation *> &newOps,
    const SetVector<Operation *> &modifiedOps,
    const SetVector<Block *> &insertedBlocks) {
  // Legalize each of the actions registered during application.
  if (failed(legalizePatternBlockRewrites(op, insertedBlocks, newOps)) ||
      failed(legalizePatternRootUpdates(modifiedOps)) ||
      failed(legalizePatternCreatedOperations(newOps))) {
    return failure();
  }
}
```

### 7. legalizePatternCreatedOperations
A failure often happens inside this function!!! <br/>
If a new op which was created through lowering some op was not marked as an legal op by the ConversionTarget instance, then a failure will happen.

For example, when lowering `ttng.async_tma_copy_global_to_local`, some ops in `mlir::gpu::GPUDialect` may be created. <br/>
If the `mlir::gpu::GPUDialect` is not marked as a legal dialect, then `legalizePatternCreatedOperations()` returns a `failure()`.
```C++
LogicalResult OperationLegalizer::legalizePatternCreatedOperations(
    const SetVector<Operation *> &newOps) {
  for (Operation *op : newOps) {
    if (failed(legalize(op))) {
      LLVM_DEBUG(logFailure(rewriter.getImpl().logger,
                            "failed to legalize generated operation '{0}'({1})",
                            op->getName(), op));
      return failure();
    }
  }
  return success();
}
```