#include "lib/Transforms/common.h"
#include "lib/Transforms/passes.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdlib> 
#include <fstream>
#include <iostream>

namespace mlir {

#define GEN_PASS_DEF_SONYDEBUGPASS
#include "lib/Transforms/tfext_kernel_gen_passes.h.inc"

namespace tfext_kernel_gen {

static std::mutex sony_mutex;

static llvm::raw_fd_ostream& getSonyOs() {
  std::error_code errCode;
  static llvm::raw_fd_ostream sonyOs("SonyDebugPass.mlir", errCode);
  if (errCode.value()) {
    std::terminate();
  }

  return sonyOs;
}

class SonyDebugPass : public impl::SonyDebugPassBase<SonyDebugPass> {
 public:
  SonyDebugPass() {}

  static bool isDone;

  void runOnOperation() override {
    if (isDone) {
      return;
    }

    const std::string kTargetGpuModuleName(std::getenv("AF_DEBUG_GPU_MODULE_NAME"));

    auto gpu_module = getOperation();
    auto context = gpu_module.getContext();
    auto module_name = gpu_module.getName();
    llvm::outs() << "gpu module: " << module_name << "\n";
    if (module_name == kTargetGpuModuleName) {
      llvm::raw_fd_ostream& sonyOs = getSonyOs();
      {
        std::lock_guard<std::mutex> lk(sony_mutex);
        sonyOs << "// SonyDebug: Entire MLIR module\n";
        sonyOs << *gpu_module->getParentOp() << "\n";
        sonyOs << "// ==========================================================================================\n\n";

        sonyOs << "// SonyDebug: Before SonyDebugPass\n";
        sonyOs << gpu_module << "\n";
        sonyOs << "// ==========================================================================================\n\n";
        sonyOs.flush();
      }

      auto& rawFuncOp = gpu_module->getRegion(0).front().front();
      OpBuilder modBuilder(gpu_module);
      modBuilder.setInsertionPoint(&rawFuncOp);

      auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(rawFuncOp);
      if (!funcOp) {
        return;
      }

      // Define types
      auto i1Type = modBuilder.getI1Type();  // bool
      auto i1PtrType = LLVM::LLVMPointerType::get(i1Type); // !llvm.ptr<i1>
      auto i8Type = IntegerType::get(context, 8);
      auto i8PtrType = LLVM::LLVMPointerType::get(i8Type);
      auto i32Type = IntegerType::get(context, 32);
      auto i32PtrType = LLVM::LLVMPointerType::get(i32Type); // !llvm.ptr<i32>
      auto f32PtrType = LLVM::LLVMPointerType::get(FloatType::getF32(context)); // !llvm.ptr<f32>

      // Define the global string constants
      // auto fmtStrType = LLVM::LLVMArrayType::get(i8Type, 17);
      auto fmtStrType = LLVM::LLVMArrayType::get(i8Type, 25);
      auto fmtStrPtrType = LLVM::LLVMPointerType::get(fmtStrType); // !llvm.ptr<!llvm.array<38 x i8>>
      // auto fmtStrAttr = modBuilder.getStringAttr("Block %d thread %d: %p[%d] -> %d | %p[%d] -> %p : %s\n");
      // auto fmtStrAttr = modBuilder.getStringAttr("==> Block %d: %s\n");
      auto fmtStrAttr = modBuilder.getStringAttr("Block %d thread %d: done\n");
      modBuilder.create<LLVM::GlobalOp>(gpu_module.getLoc(), fmtStrType, /*isConstant=*/true, LLVM::Linkage::Private, "fmt_str", fmtStrAttr);
      
      // SmallVector<Type, 3> structFieldTypes = {i32Type, i32Type, i32PtrType, i32Type, i32Type, f32PtrType, i32Type, f32PtrType, i8PtrType};
      // SmallVector<Type, 2> structFieldTypes = {i32Type, i8PtrType};
      SmallVector<Type, 2> structFieldTypes = {i32Type, i32Type};
      LLVM::LLVMStructType valistStructType = LLVM::LLVMStructType::getLiteral(context, structFieldTypes);
      LLVM::LLVMPointerType valistType = LLVM::LLVMPointerType::get(valistStructType);

      // Declare printf function
      auto printfType = LLVM::LLVMFunctionType::get(i32Type, {i8PtrType, valistType}, /*isVarArg=*/false);
      auto printfFunc = modBuilder.create<LLVM::LLVMFuncOp>(gpu_module.getLoc(), "vprintf", printfType, LLVM::Linkage::External);


      // Declare shared memory globally
      // auto doneType = LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(i1Type, 1024), 3); // Use address space 3 for CUDA shared memory
      auto blockDoneType = LLVM::LLVMArrayType::get(i1Type, 80);
      auto globalBlockDoneOp = modBuilder.create<LLVM::GlobalOp>(funcOp.getLoc(), blockDoneType, /*isConstant=*/false,
                                                                  LLVM::Linkage::Internal, "done_blocks", Attribute(), /*alignment=*/0, /*addrSpace=*/1);
      auto threadDoneType = LLVM::LLVMArrayType::get(i1Type, 512);
      auto globalThreadDoneOp = modBuilder.create<LLVM::GlobalOp>(funcOp.getLoc(), threadDoneType, /*isConstant=*/false,
                                                                  LLVM::Linkage::Internal, "done_threads", Attribute(), /*alignment=*/0, /*addrSpace=*/3);



      //===============================================================================================================
      // Serialize blocks and threads - Prologue
      //===============================================================================================================
      OpBuilder builder(funcOp);
      auto oldEntryBlock = &funcOp.getBody().front();

      Block* newEntryBlock = new Block();
      funcOp.getBody().push_front(newEntryBlock);
      for (auto& arg : oldEntryBlock->getArguments()) {
        newEntryBlock->addArgument(arg.getType(), arg.getLoc());
      }

      for (int i = 0; i < oldEntryBlock->getNumArguments(); i++) {
        oldEntryBlock->getArgument(i).replaceAllUsesWith(newEntryBlock->getArgument(i));
      }

      oldEntryBlock->eraseArguments(0, oldEntryBlock->getNumArguments());

      //============================================
      // newEntryBlock
      //============================================
      builder.setInsertionPointToStart(newEntryBlock);

      // Create constant values
      Value zero = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(0));
      Value one = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(1));
      Value two = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(2));
      Value three = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(3));
      Value four = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(4));
      Value five = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(5));
      Value six = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(6));
      Value seven = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(7));
      Value eight = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(8));
      Value nine = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(9));
      Value i1One = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i1Type, builder.getIntegerAttr(i1Type, 1));

      // Get blockIdx and threadIdx
      mlir::Value blockIdx = builder.create<mlir::NVVM::BlockIdXOp>(funcOp.getLoc(), i32Type);
      mlir::Value numBlocks = builder.create<mlir::NVVM::GridDimXOp>(funcOp.getLoc(), i32Type);
      mlir::Value threadIdx = builder.create<mlir::NVVM::ThreadIdXOp>(funcOp.getLoc(), i32Type);
      mlir::Value numThreads = builder.create<mlir::NVVM::BlockDimXOp>(funcOp.getLoc(), i32Type);

      Value blockDonePtr = builder.create<LLVM::AddressOfOp>(funcOp.getLoc(), globalBlockDoneOp);
      Value threadDonePtr = builder.create<LLVM::AddressOfOp>(funcOp.getLoc(), globalThreadDoneOp);


      //===============================================================================================================
      // Serialize blocks
      //===============================================================================================================
      Block *blockCheckDoneBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(newEntryBlock->getIterator(), blockCheckDoneBlock);

      Block *blockLoopBodyBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(blockCheckDoneBlock->getIterator(), blockLoopBodyBlock);

      Block *threadEntryBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(blockCheckDoneBlock->getIterator(), threadEntryBlock);

      Value blockGtZero = builder.create<LLVM::ICmpOp>(funcOp.getLoc(), LLVM::ICmpPredicate::ugt, blockIdx, zero);
      builder.create<LLVM::CondBrOp>(funcOp.getLoc(), blockGtZero, blockCheckDoneBlock, ValueRange{}, threadEntryBlock, ValueRange{});

      //============================================
      // 1. blockCheckDoneBlock
      //============================================
      builder.setInsertionPointToStart(blockCheckDoneBlock);
      {
        // Load done_blocks[blockIdx.x - 1]
        Value preBlockIdx = builder.create<LLVM::SubOp>(funcOp.getLoc(), i32Type, blockIdx, one);
        Value blockGep = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i1PtrType, blockDonePtr, ValueRange{zero, preBlockIdx});
        Value blockDoneVal = builder.create<LLVM::LoadOp>(funcOp.getLoc(), i1Type, blockGep);
        Value blockDone = builder.create<LLVM::AndOp>(funcOp.getLoc(), blockDoneVal, i1One);

        builder.create<LLVM::CondBrOp>(funcOp.getLoc(), blockDone, threadEntryBlock, ValueRange{}, blockLoopBodyBlock, ValueRange{});
      }

      //============================================
      // 2. blockLoopBodyBlock
      //============================================
      builder.setInsertionPointToStart(blockLoopBodyBlock);
      {
        auto voidType = LLVM::LLVMVoidType::get(context);
        auto duration = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, IntegerAttr::get(i32Type, 1000));
  
        // Define the inline assembly string for nanosleep.u32
        auto asmString = StringRef("nanosleep.u32 $0;");
  
        // Constraints: "=r" (output register), "r" (input register %r5)
        auto constraints = StringRef("r");
  
        // Dialect options: hasSideEffect=true, isAlignStack=false
        LLVM::AsmDialectAttr asmDialect = builder.getAttr<LLVM::AsmDialectAttr>(LLVM::AsmDialect::AD_ATT);
        bool hasSideEffect = true;
        bool isAlignStack = false;
        ArrayAttr operandAttrs = builder.getArrayAttr({});
  
        // Create the inline assembly operation
        builder.create<LLVM::InlineAsmOp>(
            funcOp.getLoc(),
            voidType,             // No results (void)
            ValueRange{duration}, // Input: 1000
            asmString,            // PTX instruction
            constraints,          // Input constraint for %r5
            hasSideEffect,        // Side effect: suspends execution
            isAlignStack,         // No stack alignment needed
            asmDialect,           // Use ATT dialect (though PTX-specific here)
            operandAttrs
        );
  
        // builder.create<LLVM::CallOp>(funcOp.getLoc(), TypeRange{}, nanosleepRef, ValueRange{constant});
        builder.create<LLVM::BrOp>(funcOp.getLoc(), ValueRange{}, blockCheckDoneBlock);
      }
      //===============================================================================================================


      //===============================================================================================================
      // Serialize threads
      //===============================================================================================================
      Block *initDoneBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(threadEntryBlock->getIterator(), initDoneBlock);
      Block *syncThreadsBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(initDoneBlock->getIterator(), syncThreadsBlock);
      Block *checkDoneBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(syncThreadsBlock->getIterator(), checkDoneBlock);
      Block *loopBodyBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(checkDoneBlock->getIterator(), loopBodyBlock);

      //============================================
      // 1. threadEntryBlock
      //============================================
      // threadEntryBlock->setAttr("block", builder.getStringAttr("threadEntryBlock"));
      builder.setInsertionPointToStart(threadEntryBlock);
      {
        Value gtZero = builder.create<LLVM::ICmpOp>(funcOp.getLoc(), LLVM::ICmpPredicate::sgt, threadIdx, zero);
        builder.create<LLVM::CondBrOp>(funcOp.getLoc(), gtZero, syncThreadsBlock, ValueRange{}, initDoneBlock, ValueRange{});
      }

      //============================================
      // 2. initDoneBlock
      //============================================
      builder.setInsertionPointToStart(initDoneBlock);
      {
        Value i8Zero = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i8Type, 0);
        auto threadLen = builder.create<mlir::LLVM::ConstantOp>(funcOp.getLoc(), i32Type, 512);
        auto isVolatile = builder.create<mlir::LLVM::ConstantOp>(funcOp.getLoc(), i1Type, 0);
        builder.create<mlir::LLVM::MemsetOp>(funcOp.getLoc(), threadDonePtr, i8Zero, threadLen, isVolatile);
        builder.create<LLVM::BrOp>(funcOp.getLoc(), ValueRange{}, syncThreadsBlock);
      }

      //============================================
      // 3. syncThreadsBlock
      //============================================
      builder.setInsertionPointToStart(syncThreadsBlock);
      {
        // Insert nvvm.barrier0
        builder.create<NVVM::Barrier0Op>(funcOp.getLoc()); // CUDA: __syncthreads() -> NVVM::Barrier0Op -> PTX: "bar.sync 	0;"
  
        // While loop: wait for threadIdx.x - 1 to complete
        Value isEqZero = builder.create<LLVM::ICmpOp>(funcOp.getLoc(), LLVM::ICmpPredicate::eq, threadIdx, zero);
        builder.create<LLVM::CondBrOp>(funcOp.getLoc(), isEqZero, oldEntryBlock, ValueRange{}, checkDoneBlock, ValueRange{});
      }

      //============================================
      // 4. checkDoneBlock
      //============================================
      builder.setInsertionPointToStart(checkDoneBlock);
      {
        // Load done[threadIdx.x - 1]
        Value preThreadIdx = builder.create<LLVM::SubOp>(funcOp.getLoc(), i32Type, threadIdx, one);
        Value threadGep = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i1PtrType, threadDonePtr, ValueRange{zero, preThreadIdx});
        Value threadDoneVal = builder.create<LLVM::LoadOp>(funcOp.getLoc(), i1Type, threadGep);
        Value threadDone = builder.create<LLVM::AndOp>(funcOp.getLoc(), threadDoneVal, i1One);
  
        builder.create<LLVM::CondBrOp>(funcOp.getLoc(), threadDone, oldEntryBlock, ValueRange{}, loopBodyBlock, ValueRange{});
      }

      //============================================
      // 5. loopBodyBlock
      //============================================
      builder.setInsertionPointToStart(loopBodyBlock);
      {
        auto voidType = LLVM::LLVMVoidType::get(context);
        auto duration = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, IntegerAttr::get(i32Type, 1000));
  
        // Define the inline assembly string for nanosleep.u32
        auto asmString = StringRef("nanosleep.u32 $0;");
  
        // Constraints: "=r" (output register), "r" (input register %r5)
        auto constraints = StringRef("r");
  
        // Dialect options: hasSideEffect=true, isAlignStack=false
        LLVM::AsmDialectAttr asmDialect = builder.getAttr<LLVM::AsmDialectAttr>(LLVM::AsmDialect::AD_ATT);
        bool hasSideEffect = true;
        bool isAlignStack = false;
        ArrayAttr operandAttrs = builder.getArrayAttr({});
  
        // Create the inline assembly operation
        builder.create<LLVM::InlineAsmOp>(
            funcOp.getLoc(),
            voidType,             // No results (void)
            ValueRange{duration}, // Input: 1000
            asmString,            // PTX instruction
            constraints,          // Input constraint for %r5
            hasSideEffect,        // Side effect: suspends execution
            isAlignStack,         // No stack alignment needed
            asmDialect,           // Use ATT dialect (though PTX-specific here)
            operandAttrs
        );
  
        // builder.create<LLVM::CallOp>(funcOp.getLoc(), TypeRange{}, nanosleepRef, ValueRange{constant});
        builder.create<LLVM::BrOp>(funcOp.getLoc(), ValueRange{}, checkDoneBlock);
      }
      //===============================================================================================================

      // bool visitedI32LoadOp = false;

      // // Walk through the funcOp to find the load operation
      // funcOp.walk([&](LLVM::LoadOp loadOp) {
      //   // Check if this is the load we want to modify (e.g., based on operand type)
      //   if (loadOp.getAddr().getType() == i32PtrType) {
      //     Location loc = loadOp.getLoc(); // Use the location of the load
      //     OpBuilder builder(loadOp);
      //     builder.setInsertionPointAfter(loadOp);

      //     // Create valist
      //     valist = builder.create<LLVM::AllocaOp>(funcOp.getLoc(), valistType, one, /*alignment=*/0);

      //     // Store blockIdx and threadIdx into valist
      //     Value blockMem = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i32PtrType, valist, ValueRange{zero, zero});
      //     builder.create<LLVM::StoreOp>(funcOp.getLoc(), blockIdx, blockMem);
      //     Value threadMem = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i32PtrType, valist, ValueRange{zero, one});
      //     builder.create<LLVM::StoreOp>(funcOp.getLoc(), threadIdx, threadMem);

      //     Value basePointer;
      //     Value indexVal;
      //     auto op = loadOp->getOperand(0).getDefiningOp();
      //     if (auto gepOp = dyn_cast<LLVM::GEPOp>(op)) {
      //       basePointer = gepOp.getBase();
      //       auto firstIndex = gepOp.getIndices()[0];
      //       if (firstIndex.is<mlir::Value>()) {
      //         // It’s already an mlir::Value, just extract it
      //         indexVal = firstIndex.get<mlir::Value>();
      //       } else if (firstIndex.is<mlir::IntegerAttr>()) {
      //         // It’s an IntegerAttr, convert it to an mlir::Value using a constant op
      //         mlir::IntegerAttr intAttr = firstIndex.get<mlir::IntegerAttr>();
      //         indexVal = builder.create<mlir::arith::ConstantOp>(gepOp.getLoc(), intAttr.getType(), intAttr);
      //       }
      //     } else {
      //       std::terminate();
      //     }

      //     Value i32BaseMem = builder.create<LLVM::GEPOp>(loadOp.getLoc(), i32PtrType, valist, ValueRange{zero, two});
      //     builder.create<LLVM::StoreOp>(loadOp.getLoc(), basePointer, i32BaseMem);

      //     Value i32IndexMem = builder.create<LLVM::GEPOp>(loadOp.getLoc(), i32PtrType, valist, ValueRange{zero, three});
      //     builder.create<LLVM::StoreOp>(loadOp.getLoc(), indexVal, i32IndexMem);

      //     // Get result
      //     Value i32ResMem = builder.create<LLVM::GEPOp>(loadOp.getLoc(), i32PtrType, valist, ValueRange{zero, four});
      //     builder.create<LLVM::StoreOp>(loadOp.getLoc(), loadOp.getResult(), i32ResMem);

      //     visitedI32LoadOp = true;
      //   } else if (loadOp.getAddr().getType() == f32PtrType && visitedI32LoadOp) {
      //     OpBuilder builder(loadOp);
      //     Location loc = loadOp.getLoc(); // Use the location of the load

      //     Value basePointer;
      //     Value indexVal;
      //     auto op = loadOp->getOperand(0).getDefiningOp();
      //     if (auto gepOp = dyn_cast<LLVM::GEPOp>(op)) {
      //       basePointer = gepOp.getBase();
      //       auto firstIndex = gepOp.getIndices()[0];
      //       if (firstIndex.is<mlir::Value>()) {
      //         // It’s already an mlir::Value, just extract it
      //         indexVal = firstIndex.get<mlir::Value>();
      //       } else if (firstIndex.is<mlir::IntegerAttr>()) {
      //         // It’s an IntegerAttr, convert it to an mlir::Value using a constant op
      //         mlir::IntegerAttr intAttr = firstIndex.get<mlir::IntegerAttr>();
      //         indexVal = builder.create<mlir::arith::ConstantOp>(gepOp.getLoc(), intAttr.getType(), intAttr);
      //       }
      //     } else {
      //       std::terminate();
      //     }

      //     Value f32BasePtr = builder.create<LLVM::GEPOp>(loadOp.getLoc(), f32PtrType, valist, ValueRange{zero, five});
      //     builder.create<LLVM::StoreOp>(loadOp.getLoc(), basePointer, f32BasePtr);

      //     Value i32IndexPtr = builder.create<LLVM::GEPOp>(loadOp.getLoc(), i32PtrType, valist, ValueRange{zero, six});
      //     builder.create<LLVM::StoreOp>(loadOp.getLoc(), indexVal, i32IndexPtr);

      //     // Get the pointer operand (%35)
      //     Value loadPointer = loadOp.getAddr(); // %35 in your example
      //     Value f32Ptr = builder.create<LLVM::GEPOp>(loadOp.getLoc(), f32PtrType, valist, ValueRange{zero, seven});
      //     builder.create<LLVM::StoreOp>(loadOp.getLoc(), loadPointer, f32Ptr);


      //     //=============================================================================
      //     // Get op string representation
      //     //=============================================================================
      //     std::string opStr;
      //     llvm::raw_string_ostream os(opStr);
      //     loadOp->print(os);
      //     os.flush();

      //     // Create a constant for opStr. Turn the C++ string into a MLIR SSA value.
      //     auto arrayType = LLVM::LLVMArrayType::get(i8Type, opStr.size());
      //     auto arrayConst = builder.create<LLVM::ConstantOp>(loadOp.getLoc(), arrayType, builder.getStringAttr(opStr));

      //     // Store the constant opStr into a memory location
      //     auto arrayPtrType = LLVM::LLVMPointerType::get(arrayType); // Pointer to array type
      //     auto arrayAddr = builder.create<LLVM::AllocaOp>(loadOp.getLoc(), arrayPtrType, one); // 1 element of arrayType
      //     builder.create<LLVM::StoreOp>(loadOp.getLoc(), arrayConst.getResult(), arrayAddr);

      //     // Get address of the first element of the array
      //     auto strPtr = builder.create<LLVM::GEPOp>(loadOp.getLoc(), i8PtrType, arrayAddr.getResult(), ValueRange{zero, zero});

      //     // Store the address into valist
      //     Value opStrPtr = builder.create<LLVM::GEPOp>(loadOp.getLoc(), i8PtrType, valist, ValueRange{zero, eight});
      //     builder.create<LLVM::StoreOp>(loadOp.getLoc(), strPtr.getResult(), opStrPtr);
      //     //=============================================================================

      //     // Get the address of format strings
      //     FlatSymbolRefAttr fmtStrRef = SymbolRefAttr::get(context, "fmt_str");
      //     auto fmtStrAddrOp = builder.create<LLVM::AddressOfOp>(funcOp.getLoc(), fmtStrPtrType, fmtStrRef);
      //     auto fmtStrPtr = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i8PtrType, fmtStrAddrOp.getResult(), ValueRange{zero, zero});

      //     // Create the llvm.call to printf
      //     auto callOp = builder.create<LLVM::CallOp>(loc, printfFunc, ValueRange{fmtStrPtr.getResult(), valist}); // %call

      //     visitedI32LoadOp = false;
      //   }
      // });

      //===============================================================================================================
      // Serialize blocks and threads - Epilogue
      //===============================================================================================================
      Block* oldLastBlock = &funcOp.getBody().getBlocks().back();

      Block* threadDoneBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(oldLastBlock->getIterator(), threadDoneBlock);

      Block* blockDoneBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(threadDoneBlock->getIterator(), blockDoneBlock);

      Block* blockResetDoneBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(blockDoneBlock->getIterator(), blockResetDoneBlock);

      Block* lastBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(blockResetDoneBlock->getIterator(), lastBlock);

      // Erase ReturnOp
      oldLastBlock->walk([&](LLVM::ReturnOp returnOp) {
        OpBuilder builder(returnOp);
        builder.create<LLVM::BrOp>(funcOp.getLoc(), ValueRange{}, threadDoneBlock);
        returnOp->erase();
      });

      //============================================
      // threadDoneBlock
      //============================================
      builder.setInsertionPointToStart(threadDoneBlock);
      {
        // Store done_threads[threadIdx.x] = true
        Value threadTrueVal = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i1Type, 1);
        Value threadDoneGep = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i1PtrType, threadDonePtr, ValueRange{zero, threadIdx});
        builder.create<LLVM::StoreOp>(funcOp.getLoc(), threadTrueVal, threadDoneGep);

        // Store done_blocks[blockIdx.x] = true
        Value subThreadIdx = builder.create<LLVM::SubOp>(funcOp.getLoc(), numThreads, threadIdx);
        Value isEqOne = builder.create<LLVM::ICmpOp>(funcOp.getLoc(), LLVM::ICmpPredicate::eq, subThreadIdx, one);

        builder.create<LLVM::CondBrOp>(funcOp.getLoc(), isEqOne, blockDoneBlock, ValueRange{}, lastBlock, ValueRange{});
      }

      //============================================
      // blockDoneBlock
      //============================================
      builder.setInsertionPointToStart(blockDoneBlock);
      {
        {
          auto arrayType = LLVM::LLVMArrayType::get(i8Type, 14);
          Value status = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), arrayType, builder.getStringAttr("blockDoneBlock"));
        }

        Value blockTrueVal = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i1Type, 1);
        Value blockDoneGep = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i1PtrType, blockDonePtr, ValueRange{zero, blockIdx});
        builder.create<LLVM::StoreOp>(funcOp.getLoc(), blockTrueVal, blockDoneGep);

        Value subBlockIdx = builder.create<LLVM::SubOp>(funcOp.getLoc(), numBlocks, blockIdx);
        Value isEqOne = builder.create<LLVM::ICmpOp>(funcOp.getLoc(), LLVM::ICmpPredicate::eq, subBlockIdx, one);

        builder.create<LLVM::CondBrOp>(funcOp.getLoc(), isEqOne, blockResetDoneBlock, ValueRange{}, lastBlock, ValueRange{});
      }

      //============================================
      // blockResetDoneBlock
      //============================================
      builder.setInsertionPointToStart(blockResetDoneBlock);
      {
        Value i8Zero = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i8Type, 0);
        auto blockLen = builder.create<mlir::LLVM::ConstantOp>(funcOp.getLoc(), i32Type, 80);
        auto isVolatile = builder.create<mlir::LLVM::ConstantOp>(funcOp.getLoc(), i1Type, 0);
        builder.create<mlir::LLVM::MemsetOp>(funcOp.getLoc(), blockDonePtr, i8Zero, blockLen, isVolatile);

        builder.create<LLVM::BrOp>(funcOp.getLoc(), ValueRange{}, lastBlock);
      }

      //============================================
      // lastBlock
      //============================================
      builder.setInsertionPointToStart(lastBlock);
      {
        // Create valist
        Value valist = builder.create<LLVM::AllocaOp>(funcOp.getLoc(), valistType, one, /*alignment=*/0);

        // Store blockIdx and threadIdx into valist
        Value blockMem = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i32PtrType, valist, ValueRange{zero, zero});
        builder.create<LLVM::StoreOp>(funcOp.getLoc(), blockIdx, blockMem);
        Value threadMem = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i32PtrType, valist, ValueRange{zero, one});
        builder.create<LLVM::StoreOp>(funcOp.getLoc(), threadIdx, threadMem);

        // Get the address of format strings
        FlatSymbolRefAttr fmtStrRef = SymbolRefAttr::get(context, "fmt_str");
        auto fmtStrAddrOp = builder.create<LLVM::AddressOfOp>(funcOp.getLoc(), fmtStrPtrType, fmtStrRef);
        auto fmtStrPtr = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i8PtrType, fmtStrAddrOp.getResult(), ValueRange{zero, zero});

        // Create the llvm.call to printf
        auto callOp = builder.create<LLVM::CallOp>(funcOp.getLoc(), printfFunc, ValueRange{fmtStrPtr.getResult(), valist}); // %call

        builder.create<LLVM::ReturnOp>(funcOp.getLoc(), ValueRange{});
      }
      //===============================================================================================================

      {
        std::lock_guard<std::mutex> lk(sony_mutex);
        sonyOs << "// SonyDebug: After SonyDebugPass\n";
        sonyOs << gpu_module << "\n";
        sonyOs << "// ==========================================================================================\n\n";
        sonyOs.flush();
      }

      SonyDebugPass::isDone = true;
    }
  }
};

bool SonyDebugPass::isDone = false;

// std::unique_ptr<OperationPass<func::FuncOp>> createSonyDebugPass() {
//   return std::make_unique<SonyDebugPass>();
// }

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createSonyDebugPass() {
  return std::make_unique<SonyDebugPass>();
}

} // tfext_kernel_gen
} // mlir


// Status GenerateDeviceCode(mlir::ModuleOp module,
//                           llvm::StringRef gpu_binary_attr_name,
//                           llvm::ArrayRef<std::string> architectures,
//                           bool print_ptx, bool print_llvmir, bool enable_ftz,
//                           bool apply_cl_options,
//                           mlir::DefaultTimingManager &tm) {
//   auto ctx = module.getContext();
//   ctx->enableMultithreading(false);
//   mlir::PassManager pm(ctx);
//   auto timing = tm.getRootScope();
//   pm.enableTiming(timing);
//   if (apply_cl_options)
//     applyTensorflowAndCLOptions(pm);
//   mlir::registerLLVMDialectTranslation(*module->getContext());

//   auto &kernel_pm = pm.nest<mlir::gpu::GPUModuleOp>();

//   kernel_pm.addPass(mlir::tfext_kernel_gen::createSonyDebugPass());

//   // Remove debug information to ensure we do not create debug PTX.
//   kernel_pm.addPass(mlir::createStripDebugInfoPass());
//   kernel_pm.addPass(mlir::kernel_gen::transforms::CreateGpuKernelToBlobPass(
//       gpu_binary_attr_name, architectures, print_ptx, print_llvmir,
//       enable_ftz));

//   return failed(pm.run(module))
//              ? tensorflow::errors::Internal("Generating device code failed.")
//              : OkStatus();
// }

static std::mutex sony_mutex;

static llvm::raw_fd_ostream &getSonyOs(const char *fileName) {
  std::error_code errCode;
  static llvm::raw_fd_ostream sonyOs(fileName, errCode);
  if (errCode.value()) {
    std::terminate();
  }

  return sonyOs;
}