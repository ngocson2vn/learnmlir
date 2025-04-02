        // Atomic add operation
        for (int i = 0; i < inputs.size(); i++) {
          Value partialSumVal = rewriter.create<memref::LoadOp>(loc, partialSumBuffer, indexValues[i]);
          if (partialSumVal.getType().isa<IntegerType>()) {
            rewriter.create<memref::AtomicRMWOp>(
              loc,
              arith::AtomicRMWKind::addi, // Integer add
              partialSumVal,              // Value to add
              sharedAccBuffer,            // Memref to update
              ValueRange{indexValues[i]}  // Index
            );
          } else if (partialSumVal.getType().isa<FloatType>()) {
            rewriter.create<memref::AtomicRMWOp>(
              loc,
              arith::AtomicRMWKind::addf, // Floating-point add
              partialSumVal,              // Value to add
              sharedAccBuffer,            // Memref to update
              ValueRange{indexValues[i]}  // Index
            );
          } else {
            llvm::errs() << "Unsupported data type\n";
            std::terminate();
          }
        }