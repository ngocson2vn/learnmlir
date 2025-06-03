bool isValidCluster(Cluster& c) const {
  auto cluster = c.getCluster();
  llvm::DenseSet<Operation *> ops(cluster.begin(), cluster.end());
  bool foundTargetOp = false;
  int targetOpUserCount = 0;
  for (auto op : cluster) {
    auto innerOp = islandOpInnerOp(op);
    if (isa<TF::FillOp, TF::ReshapeOp>(innerOp)) {
      foundTargetOp = true;
      auto islandOp = cast<tf_executor::IslandOp>(op);
      for (auto res : islandOp.getOutputs()) {
        for (auto user : res.getUsers()) {
          if (ops.count(user->getParentOp())) {
            targetOpUserCount++;
          }
        }
      }
    }
  }

  if (foundTargetOp && targetOpUserCount == 0) {
    return false;
  }

  return true;
}