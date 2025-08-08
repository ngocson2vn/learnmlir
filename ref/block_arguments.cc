int numArgs = funcOp.getNumArguments();
Block& block = funcOp.getFunctionBody().front();
for (int i = 0; i < numArgs; i++) {
  auto oldArg = block.getArgument(i);
  block.addArgument(newArgTypes[i], oldArg.getLoc());
}

for (int i = 0; i < numArgs; i++) {
  auto oldArg = block.getArgument(i);
  oldArg.replaceAllUsesWith(block.getArgument(numArgs + i));
}

// Remove 0 ~ numArgs -1 arguments
block.eraseArguments(0, numArgs);