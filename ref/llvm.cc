// Get operation name
// llvm/include/llvm/CodeGen/SelectionDAGNodes.h
{
  if (Node->getNumOperands() > 0) {
    auto srcType = Node->getOperand(0).getValueType();
    auto dstType = Node->getValueType(0);
    if (srcType == MVT::i32 && dstType == MVT::bf16) {
      llvm::outs() << Node->getOperationName() << ": " << Node->getOpcode() << "\n";
    }
  }
}