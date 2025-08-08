struct TensorToMemRefConverter : public TypeConverter {
  TensorToMemRefConverter() {
    // Map tensor to memref
    addConversion([](TensorType tensor) -> Type {
      if (auto rankedTensor = dyn_cast<RankedTensorType>(tensor)) {
        return MemRefType::get(rankedTensor.getShape(), rankedTensor.getElementType());
      }
      return tensor;
    });

    // Allow memref types to pass through unchanged
    addConversion([](MemRefType memref) { return memref; });
  }
};

    // // Register source materialization: memref<?xf64> -> tensor<?xf64>
    // addSourceMaterialization(
    //     [](mlir::OpBuilder &builder, mlir::Type resultType,
    //        mlir::ValueRange convertedValues, mlir::Location loc) -> mlir::Value {
    //       assert(convertedValues.size() == 1 && "convertedValues must have size = 1");
    //       auto srcValue = builder.create<mlir::bufferization::ToTensorOp>(loc, resultType, convertedValues[0]);
    //       return srcValue;
    //     });

    // // Register target materialization: tensor<?xf64> -> memref<?xf64>
    // addTargetMaterialization(
    //     [](mlir::OpBuilder &builder, mlir::TypeRange resultTypes,
    //        mlir::ValueRange srcValues, mlir::Location loc) -> SmallVector<Value> {

    //       SmallVector<Value> tgtValues;
    //       for (const auto& [t, v] : llvm::zip(resultTypes, srcValues)) {
    //         auto ret = builder.create<mlir::bufferization::ToBufferOp>(loc, t, v);
    //         llvm::outs() << "\n" << ret << "\n";
    //         tgtValues.push_back(ret);
    //       }

    //       return tgtValues;
    //     });