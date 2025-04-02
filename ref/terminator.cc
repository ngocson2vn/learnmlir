if (!thenBlock.empty() && isa<mlir::scf::YieldOp>(thenBlock.back())) {
  thenBlock.back().erase();
}