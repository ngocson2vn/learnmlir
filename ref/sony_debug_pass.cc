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
  static llvm::raw_fd_ostream sonyOs("/data00/home/son.nguyen/workspace/auto_fusion_dev/AutoFusion/bto/debug_v3/SonyDebugPass.mlir", errCode);
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

    auto gpu_module = getOperation();
    auto context = gpu_module.getContext();
    auto module_name = gpu_module.getName();
    llvm::outs() << "gpu module: " << module_name << "\n";
    if (module_name == "predict_online_1_kernel") {
      llvm::raw_fd_ostream& sonyOs = getSonyOs();
      {
        std::lock_guard<std::mutex> lk(sony_mutex);
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
      
      SmallVector<Type, 3> fieldTypes = {i32Type, i32Type, f32PtrType, i8PtrType};
      LLVM::LLVMStructType structType = LLVM::LLVMStructType::getLiteral(context, fieldTypes);
      LLVM::LLVMPointerType valistType = LLVM::LLVMPointerType::get(structType);

      // Define the global string constant
      auto i8ArrayType = LLVM::LLVMArrayType::get(i8Type, 35);
      auto stringAttr = modBuilder.getStringAttr("Block %d thread %d pointer %p : %s\n");
      modBuilder.create<LLVM::GlobalOp>(gpu_module.getLoc(), i8ArrayType, /*isConstant=*/true, LLVM::Linkage::Private, "fmt_str", stringAttr);

      // Declare printf
      auto printfTy = LLVM::LLVMFunctionType::get(i32Type, {i8PtrType, valistType});
      modBuilder.create<LLVM::LLVMFuncOp>(gpu_module.getLoc(), "vprintf", printfTy, LLVM::Linkage::External);

      // Declare shared memory globally
      // auto doneType = LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(i1Type, 1024), 3); // Use address space 3 for CUDA shared memory
      auto doneType = LLVM::LLVMArrayType::get(i1Type, 1024);
      auto globalDoneOp = modBuilder.create<LLVM::GlobalOp>(funcOp.getLoc(), doneType, /*isConstant=*/false,
                                                            LLVM::Linkage::Internal, "done", Attribute(), /*alignment=*/0, /*addrSpace=*/3);

      //===============================================================================================================
      // Serialize threads - Prologue
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
      // 1. newEntryBlock
      //============================================
      builder.setInsertionPointToStart(newEntryBlock);

      // Create the nvvm.read.ptx.sreg.tid.x operation
      mlir::Value blockIdx = builder.create<mlir::NVVM::BlockIdXOp>(funcOp.getLoc(), i32Type);
      mlir::Value threadIdx = builder.create<mlir::NVVM::ThreadIdXOp>(funcOp.getLoc(), i32Type);
      Value zero = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(0));
      Value one = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(1));
      Value two = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(2));
      Value three = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(3));

      // Create valist
      Value valist = builder.create<LLVM::AllocaOp>(funcOp.getLoc(), valistType, one, /*alignment=*/0);
      Value blockPtr = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i32PtrType, valist, ValueRange{zero, zero});
      Value threadPtr = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i32PtrType, valist, ValueRange{zero, one});
      builder.create<LLVM::StoreOp>(funcOp.getLoc(), blockIdx, blockPtr);
      builder.create<LLVM::StoreOp>(funcOp.getLoc(), threadIdx, threadPtr);

      Value donePtr = builder.create<LLVM::AddressOfOp>(funcOp.getLoc(), globalDoneOp);
      Value gtZero = builder.create<LLVM::ICmpOp>(funcOp.getLoc(), LLVM::ICmpPredicate::sgt, threadIdx, zero);

      Block *initDoneBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(newEntryBlock->getIterator(), initDoneBlock);
      Block *syncThreadsBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(initDoneBlock->getIterator(), syncThreadsBlock);
      builder.create<LLVM::CondBrOp>(funcOp.getLoc(), gtZero, syncThreadsBlock, ValueRange{}, initDoneBlock, ValueRange{});

      //============================================
      // 2. initDoneBlock
      //============================================
      builder.setInsertionPointToStart(initDoneBlock);
      Value i8Zero = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i8Type, 0);
      auto len = builder.create<mlir::LLVM::ConstantOp>(funcOp.getLoc(), i32Type, 1024);
      auto isVolatile = builder.create<mlir::LLVM::ConstantOp>(funcOp.getLoc(), i1Type, 0);
      builder.create<mlir::LLVM::MemsetOp>(funcOp.getLoc(), donePtr, i8Zero, len, isVolatile);
      builder.create<LLVM::BrOp>(funcOp.getLoc(), ValueRange{}, syncThreadsBlock);

      //============================================
      // 3. syncThreadsBlock
      //============================================
      builder.setInsertionPointToStart(syncThreadsBlock);

      // Insert nvvm.barrier0
      builder.create<NVVM::Barrier0Op>(funcOp.getLoc()); // CUDA: __syncthreads() -> NVVM::Barrier0Op -> PTX: "bar.sync 	0;"

      // While loop: wait for threadIdx.x - 1 to complete
      Block *checkDoneBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(syncThreadsBlock->getIterator(), checkDoneBlock);
      Value isEqZero = builder.create<LLVM::ICmpOp>(funcOp.getLoc(), LLVM::ICmpPredicate::eq, threadIdx, zero);
      builder.create<LLVM::CondBrOp>(funcOp.getLoc(), isEqZero, oldEntryBlock, ValueRange{}, checkDoneBlock, ValueRange{});

      //============================================
      // 4. checkDoneBlock
      //============================================
      builder.setInsertionPointToStart(checkDoneBlock);

      // Load done[threadIdx.x - 1]
      Value predIdx = builder.create<LLVM::SubOp>(funcOp.getLoc(), i32Type, threadIdx, one);
      Value gep = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i1PtrType, donePtr, ValueRange{zero, predIdx});
      Value doneVal = builder.create<LLVM::LoadOp>(funcOp.getLoc(), i1Type, gep);
      Value i1One = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i1Type, builder.getIntegerAttr(i1Type, 1));
      Value notDone = builder.create<LLVM::XOrOp>(funcOp.getLoc(), doneVal, i1One);
      
      Block *loopBodyBlock = new Block();
      funcOp.getBody().getBlocks().insertAfter(checkDoneBlock->getIterator(), loopBodyBlock);

      builder.create<LLVM::CondBrOp>(funcOp.getLoc(), notDone, loopBodyBlock, ValueRange{}, oldEntryBlock, ValueRange{});

      //============================================
      // 5. loopBodyBlock
      //============================================
      builder.setInsertionPointToStart(loopBodyBlock);
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
      //===============================================================================================================

      // Walk through the funcOp to find the load operation
      funcOp.walk([&](LLVM::LoadOp loadOp) {
        // Check if this is the load we want to modify (e.g., based on operand type)
        if (loadOp.getAddr().getType() == f32PtrType) {
          OpBuilder builder(loadOp);
          Location loc = loadOp.getLoc(); // Use the location of the load

          // Get the pointer operand (%35)
          Value pointer = loadOp.getAddr(); // %35 in your example

          // Get the address of format string
          FlatSymbolRefAttr fmtStrRef = SymbolRefAttr::get(context, "fmt_str");
          auto i8ArrayPtrType = LLVM::LLVMPointerType::get(i8ArrayType); // !llvm.ptr<!llvm.array<20 x i8>>
          auto addrOp = builder.create<LLVM::AddressOfOp>(loadOp.getLoc(), i8ArrayPtrType, fmtStrRef);
          auto zero = builder.create<LLVM::ConstantOp>(loadOp.getLoc(), i32Type, 0);
          SmallVector<Value> indices = {zero, zero}; // [0, 0] to point to the first element
          auto gepOp = builder.create<LLVM::GEPOp>(loadOp.getLoc(), i8PtrType, addrOp.getResult(), indices);

          // auto op = loadOp->getOperand(0).getDefiningOp();
          // Value pointer;
          // Value index;
          // if (auto gepOp = dyn_cast<LLVM::GEPOp>(op)) {
          //   pointer = gepOp.getBase();
          //   auto firstIndex = gepOp.getIndices()[0];
          //   if (firstIndex.is<mlir::Value>()) {
          //     // It’s already an mlir::Value, just extract it
          //     index = firstIndex.get<mlir::Value>();
          //   } else if (firstIndex.is<mlir::IntegerAttr>()) {
          //     // It’s an IntegerAttr, convert it to an mlir::Value using a constant op
          //     mlir::IntegerAttr intAttr = firstIndex.get<mlir::IntegerAttr>();
          //     index = builder.create<mlir::arith::ConstantOp>(gepOp.getLoc(), intAttr.getType(), intAttr);
          //   }
          // } else {
          //   return;
          // }

          Value f32Ptr = builder.create<LLVM::GEPOp>(loadOp.getLoc(), f32PtrType, valist, ValueRange{zero, two});
          builder.create<LLVM::StoreOp>(loadOp.getLoc(), pointer, f32Ptr);

          std::string opStr;
          llvm::raw_string_ostream os(opStr);
          loadOp->print(os);
          os.flush();

          // Create a constant for opStr. Turn the C++ string into a MLIR SSA value.
          auto arrayType = LLVM::LLVMArrayType::get(i8Type, opStr.size());
          auto arrayConst = builder.create<LLVM::ConstantOp>(loadOp.getLoc(), arrayType, builder.getStringAttr(opStr));

          // Store the constant opStr into a memory location
          auto arrayPtrType = LLVM::LLVMPointerType::get(arrayType); // Pointer to array type
          auto arrayAddr = builder.create<LLVM::AllocaOp>(loadOp.getLoc(), arrayPtrType, one); // 1 element of arrayType
          builder.create<LLVM::StoreOp>(loadOp.getLoc(), arrayConst.getResult(), arrayAddr);

          // Get address of the first element of the array
          auto strPtr = builder.create<LLVM::GEPOp>(loadOp.getLoc(), i8PtrType, arrayAddr.getResult(), ValueRange{zero, zero});

          // Store the address into valist
          Value opStrPtr = builder.create<LLVM::GEPOp>(loadOp.getLoc(), i8PtrType, valist, ValueRange{zero, three});
          builder.create<LLVM::StoreOp>(loadOp.getLoc(), strPtr, opStrPtr);

          // Declare or reference the printf function
          FlatSymbolRefAttr printfRef = SymbolRefAttr::get(context, "vprintf");

          // Create the llvm.call to printf
          auto callOp = builder.create<LLVM::CallOp>(loc, i32Type, printfRef, ValueRange{gepOp.getResult(), valist}); // %call
        }
      });

      //===============================================================================================================
      // Serialize threads - Epilogue
      //===============================================================================================================
      funcOp.walk([&](LLVM::ReturnOp returnOp) {
        OpBuilder builder(returnOp);
        // Store done[threadIdx.x] = true
        Value donePtr = builder.create<LLVM::AddressOfOp>(funcOp.getLoc(), globalDoneOp);
        Value trueVal = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i1Type, 1);
        Value doneGep = builder.create<LLVM::GEPOp>(funcOp.getLoc(), i1PtrType, donePtr, ValueRange{zero, threadIdx});
        builder.create<LLVM::StoreOp>(funcOp.getLoc(), trueVal, doneGep);
      });
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
