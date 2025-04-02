Value blockStepBuffer = rewriter.create<memref::AllocaOp>(loc, MemRefType::get({}, rewriter.getIndexType()));
rewriter.create<memref::StoreOp>(loc, c1, blockStepBuffer);
Value blockStep = rewriter.create<memref::LoadOp>(loc, blockStepBuffer);