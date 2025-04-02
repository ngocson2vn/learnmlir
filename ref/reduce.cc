      // // Nested scf.parallel for the k-th dimension with reduction
      // Location reduceParLoc = NameLoc::get(StringAttr::get(context, "Innermost ParallelOp"));
      // auto reducePar = builder.create<scf::ForOp>(reduceParLoc, ValueRange{c0}, ValueRange{dims[k]}, ValueRange{c1}, ValueRange{c0_i32});
      // {
      //   OpBuilder::InsertionGuard guard(builder);
      //   builder.setInsertionPointToStart(reducePar.getBody());
      //   inIndices[k] = reducePar.getInductionVars()[0];

      //   // Load value from inputTensor
      //   Value val = builder.create<memref::LoadOp>(loc, inputTensor, inIndices);

      //   // Reduce
      //   auto reduce = builder.create<scf::ReduceOp>(loc, val);
      //   {
      //     OpBuilder::InsertionGuard guard(builder);
      //     builder.setInsertionPointToStart(&reduce.getReductionOperator().front());
      //     Value lhs = reduce.getReductionOperator().getArgument(0);  // i32 (accumulator)
      //     Value rhs = reduce.getReductionOperator().getArgument(1);  // i32 (current value)
      //     Value sum = builder.create<arith::AddIOp>(loc, lhs, rhs);
      //     builder.create<scf::ReduceReturnOp>(loc, sum);
      //   }
      // }


        // } else if (nestedLevel == 1) { // Thread level loop
          // Value numThreads = launchOp.getBlockSizeX();
          // Value estThreadStep =
          //     rewriter.create<arith::DivUIOp>(loc, yDimSize, numThreads);
          // Value threadStepBuffer = rewriter.create<memref::AllocaOp>(
          //     loc, MemRefType::get({}, rewriter.getIndexType()));

          // // Default threadStep = 1
          // rewriter.create<memref::StoreOp>(loc, c1, threadStepBuffer);

          // // Check if estThreadStep > 0
          // Value threadStepCond = rewriter.create<arith::CmpIOp>(
          //     loc, arith::CmpIPredicate::ugt, estThreadStep, c0);
          // auto threadIfOp =
          //     rewriter.create<scf::IfOp>(loc, threadStepCond, false);
          // {
          //   OpBuilder::InsertionGuard guard(rewriter);
          //   Block *thenBlock = threadIfOp.thenBlock();
          //   rewriter.setInsertionPointToStart(thenBlock);
          //   rewriter.create<memref::StoreOp>(loc, estThreadStep,
          //                                    threadStepBuffer);
          // }

          // Value threadStep =
          //     rewriter.create<memref::LoadOp>(loc, threadStepBuffer);
          // Value threadIdx = rewriter.create<gpu::ThreadIdOp>(
          //     loc, rewriter.getIndexType(), gpu::Dimension::x);
          // Value threadLb =
          //     rewriter.create<arith::MulIOp>(loc, threadIdx, threadStep);
          // Value estThreadUb =
          //     rewriter.create<arith::AddIOp>(loc, threadLb, threadStep);
          // Value remThreads =
          //     rewriter.create<arith::RemUIOp>(loc, yDimSize, numThreads);
          // Value maxThreadUb =
          //     rewriter.create<arith::AddIOp>(loc, estThreadUb, remThreads);
          // Value threadIdxDelta =
          //     rewriter.create<arith::SubIOp>(loc, numThreads, threadIdx);
          // Value threadUbCond = rewriter.create<arith::CmpIOp>(
          //     loc, arith::CmpIPredicate::ugt, threadIdxDelta, c1);
          // Value threadUb = rewriter.create<arith::SelectOp>(
          //     loc, threadUbCond, estThreadUb, maxThreadUb);

          // auto threadForLoc = NameLoc::get(
          //     StringAttr::get(context, funcOp.getSymName() + " threadForOp"));
          // auto threadForOp =
          //     rewriter.create<scf::ForOp>(threadForLoc, threadLb, threadUb, c1);
          // rewriter.setInsertionPointToStart(threadForOp.getBody());
          // inIndices[i] = threadForOp.getInductionVar();
          // outIndices.push_back(threadForOp.getInductionVar());


    // // Calculate dynamic shared memory size
    // Value numSharedElements = rewriter.create<arith::MinUIOp>(loc, dims[k], numThreads);
    // Value sharedMemBytes;
    // if (inputType.getElementType().isa<decltype(IntegerType::get(context, 32))>()) {          // i32
    //   sharedMemBytes = rewriter.create<arith::MulIOp>(loc, numSharedElements, c4);
    // } else if (inputType.getElementType().isa<decltype(IntegerType::get(context, 64))>()) {   // i64
    //   sharedMemBytes = rewriter.create<arith::MulIOp>(loc, numSharedElements, c8);
    // } else if (inputType.getElementType().isa<decltype(FloatType::getF16(context))>()) {      // f16
    //   sharedMemBytes = rewriter.create<arith::MulIOp>(loc, numSharedElements, c2);
    // } else if (inputType.getElementType().isa<decltype(FloatType::getBF16(context))>()) {     // bf16
    //   sharedMemBytes = rewriter.create<arith::MulIOp>(loc, numSharedElements, c2);
    // } else if (inputType.getElementType().isa<decltype(FloatType::getF32(context))>()) {      // f32
    //   sharedMemBytes = rewriter.create<arith::MulIOp>(loc, numSharedElements, c4);
    // } else {
    //   llvm::errs() << "Unsupported data type\n";
    //   std::terminate();
    // }