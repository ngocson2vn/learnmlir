struct GenericOpConverter : public ConversionPattern {
  GenericOpConverter(mlir::TypeConverter &typeConverter,
                     mlir::MLIRContext *context,
                     mlir::PatternBenefit benefit = 1)
      : mlir::ConversionPattern(typeConverter, mlir::Pattern::MatchAnyOpTypeTag{},
                                benefit, context) {}

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const override {
    // Skip operations that are already legal (e.g., func.func, return)
    if (op->getName().getDialectNamespace() == BuiltinDialect::getDialectNamespace() || op->getName().getStringRef() == func::FuncOp::getOperationName()) {
      return failure();
    }

    // Convert result types
    SmallVector<Type, 1> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), newResultTypes))) {
      return failure();
    }

    // Check if the operation is type-polymorphic and can handle memrefs
    bool needsConversion = false;
    for (Type resultType : op->getResultTypes()) {
      if (isa<TensorType>(resultType)) {
        needsConversion = true;
        break;
      }
    }
    for (Value operand : op->getOperands()) {
      if (isa<TensorType>(operand.getType())) {
        needsConversion = true;
        break;
      }
    }

    if (!needsConversion) {
      return failure(); // No tensor types to convert
    }

    // Create a new operation with converted types
    OperationState newState(op->getLoc(), op->getName(), operands, newResultTypes, op->getAttrs());
    Operation *newOp = rewriter.create(newState);

    // Replace the original operation
    auto opName = op->getName().getStringRef();
    auto parentOp = op->getParentOp();
    rewriter.replaceOp(op, newOp->getResults());

    llvm::outs() << "GenericOpConverter: op = " << opName << ":\n";
    llvm::outs() << *parentOp << "\n\n";
    return success();
  }
};