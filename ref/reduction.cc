        //============================================================================================
        // Tree-based parallel reduction
        //============================================================================================
        Value initVal = rewriter.create<arith::DivUIOp>(loc, numThreads, c2);
        auto whileOp = rewriter.create<scf::WhileOp>(loc, TypeRange{indexType}, ValueRange{initVal});

        // Condition region: check if iv > 0
        {
          OpBuilder::InsertionGuard guard(rewriter);
          Block &condBlock = whileOp.getBefore().emplaceBlock();
          condBlock.addArguments({indexType}, {loc}); // iv
          rewriter.setInsertionPointToStart(&condBlock);

          Value iv = condBlock.getArgument(0);
          Value ok = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, iv, c0);
          rewriter.create<scf::ConditionOp>(loc, ok, iv);
        }

        // Body region
        {
          OpBuilder::InsertionGuard guard(rewriter);
          Block &bodyBlock = whileOp.getAfter().emplaceBlock();
          bodyBlock.addArguments({indexType}, {loc}); // iv
          rewriter.setInsertionPointToStart(&bodyBlock);

          Value iv = bodyBlock.getArgument(0);
          Value threadCond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, threadIdx, iv);
          auto ifOp = rewriter.create<scf::IfOp>(loc, threadCond, false);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ifOp.thenBlock());

            // // DEBUG
            // // =========================================================================================================
            // Value i64Iv = rewriter.create<arith::IndexCastOp>(loc, i64Type, iv);
            // rewriter.create<LLVM::StoreOp>(loc, i64Iv, indexMem);
            // rewriter.create<LLVM::CallOp>(loc, printfFunc, ValueRange{fmtStrPtr.getResult(), valist});
            // // =========================================================================================================

            Value nextIndex = rewriter.create<arith::AddIOp>(loc, threadIdx, iv);
            Value v1 = rewriter.create<memref::LoadOp>(loc, sharedAccBuffer, ValueRange{threadIdx});
            Value v2 = rewriter.create<memref::LoadOp>(loc, sharedAccBuffer, ValueRange{nextIndex});
            Value accVal;
            if (v1.getType().isa<IntegerType>()) {
              accVal = rewriter.create<arith::AddIOp>(loc, v1, v2);
            } else if (v1.getType().isa<FloatType>()) {
              accVal = rewriter.create<arith::AddFOp>(loc, v1, v2);
            }

            rewriter.create<memref::StoreOp>(loc, accVal, sharedAccBuffer, ValueRange{threadIdx});
          }

          // Synchronize threads to ensure that
          // all threads have done computing its partial sum
          rewriter.create<gpu::BarrierOp>(loc);

          // Update iv: iv = iv >> 1 (divided by 2)
          Value newIv = rewriter.create<arith::ShRUIOp>(loc, iv, c1);

          // Yield the new iv
          rewriter.create<scf::YieldOp>(loc, ValueRange{newIv});
        }
        //============================================================================================

        // Thread 0 stores accumulated result back to global memory
        Value thread0 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, threadIdx, c0);
        auto ifOp = rewriter.create<scf::IfOp>(loc, thread0, false);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(ifOp.thenBlock());

          // Compute output index
          Value outputIndexVal = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(0));
          for (int i = 0; i < outputIndices.size(); i++) {
            Value indexVal = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(i));

            Value outStrideIndexVal = rewriter.create<LLVM::AddOp>(loc, indexVal, outSSBaseIndex);
            Value outStridePtr = rewriter.create<LLVM::GEPOp>(loc, outStrideArrayType, devOutStrideArrayPtr, ValueRange{outStrideIndexVal});
            Value outStrideVal = rewriter.create<LLVM::LoadOp>(loc, i64Type, outStridePtr);

            Value tmpIndexVal = rewriter.create<LLVM::MulOp>(loc, outputIndices[i], outStrideVal);
            outputIndexVal = rewriter.create<LLVM::AddOp>(loc, tmpIndexVal, outputIndexVal);
          }

          auto outputElementPtrType = outputTensorPtr.getType().cast<LLVM::LLVMPointerType>();
          Value outputElementPtr = rewriter.create<LLVM::GEPOp>(loc, outputElementPtrType, outputTensorPtr, ValueRange{outputIndexVal});
          Value finalAccVal = rewriter.create<memref::LoadOp>(loc, sharedAccBuffer, ValueRange{c0});
          rewriter.create<LLVM::StoreOp>(loc, finalAccVal, outputElementPtr);
        }