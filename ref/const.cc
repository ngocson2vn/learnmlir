if (auto constOp = dyn_cast<TF::ConstOp>(islandOpInnerOp(defOp))) {
  auto denseAttr = constOp.value().cast<DenseElementsAttr>();
  // Currently, support 1D only
  auto index = denseAttr.getValues<int32_t>()[0];
  rdIndicesAttr = b.getI32IntegerAttr(index);
} else {
  llvm::errs() << "error: reduction_indices is not defined by a tf.Const op\n";
  std::terminate();
}