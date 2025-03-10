/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//===- kernel_creator.cc ----------------------------------------*- C++ -*-===//
//
// This file implements the function to compile a TF kernel function to gpu
// binary (hsaco for AMD, cubin for NVIDIA) or to a gpu binary with host side.
//
//===----------------------------------------------------------------------===//
#include "kernel_creator.h"

#include <string>

#include "passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/OptimizeForNVVM.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/ChloOps.h" // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Transforms/gpu_passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Transforms/passes.h"
#include "tensorflow/core/platform/statusor.h"
#include <ostream>

#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"  // from @llvm-project

namespace tensorflow {
namespace kernel_gen {
namespace {

using mlir::Value;
using mlir::func::FuncOp;
using mlir::memref::RankOp;

constexpr llvm::StringRef kGpuBinaryAttrName = "gpu.binary";

/// Check if the size of the allocation is less than the given size. The
/// transformation is only applied to small buffers since large buffers could
/// exceed the stack space.
bool IsSmallAlloc(Value alloc) {
  constexpr unsigned kMaximumSizeInBytes = 64;
  constexpr unsigned kMaxRankOfAllocatedMemRef = 1;

  auto type = alloc.getType().dyn_cast<mlir::ShapedType>();
  if (!type || !alloc.getDefiningOp<mlir::memref::AllocOp>()) {
    return false;
  }
  auto memref = alloc.getType().cast<mlir::MemRefType>();
  if (memref.getMemorySpaceAsInt() != 0) {
    return true;
  }
  if (!type.hasStaticShape()) {
    // Check if the dynamic shape dimension of the alloc is produced by RankOp
    // or SelectOp(_, RankOp, RankOp).
    // If this is the case, it is likely to be small. Furthermore, the dimension
    // is limited to the maximum rank of the allocated memref to avoid large
    // values by multiplying several small values.
    if (type.getRank() <= kMaxRankOfAllocatedMemRef) {
      for (Value alloc_arg : alloc.getDefiningOp()->getOperands()) {
        if (auto select = alloc_arg.getDefiningOp<mlir::arith::SelectOp>()) {
          if (!select.getTrueValue().getDefiningOp<RankOp>() ||
              !select.getFalseValue().getDefiningOp<RankOp>())
            return false;
        } else if (!alloc_arg.getDefiningOp<RankOp>()) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
  unsigned bitwidth = mlir::DataLayout::closest(alloc.getDefiningOp())
                          .getTypeSizeInBits(type.getElementType());
  return type.getNumElements() * bitwidth <= kMaximumSizeInBytes * 8;
}

Status LowerTFToJITInvocation(mlir::ModuleOp module,
                              llvm::ArrayRef<int64_t> tile_sizes,
                              llvm::ArrayRef<int64_t> unroll_factors,
                              int64_t max_supported_rank, bool enable_ftz,
                              bool index_64bit,
                              bool jit_i64_indexed_for_large_tensors,
                              bool apply_cl_options) {
  mlir::PassManager pm(module.getContext());
  if (apply_cl_options)
    applyTensorflowAndCLOptions(pm);

  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateTFToJITInvocationPass(
          tile_sizes, unroll_factors, max_supported_rank, enable_ftz,
          index_64bit, jit_i64_indexed_for_large_tensors));
  pm.addPass(mlir::kernel_gen::tf_framework::CreateEmbedTFFrameworkPass());
  pm.addNestedPass<FuncOp>(
      mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(mlir::createComputeOpAndFuncBufferizePass());

  pm.addPass(mlir::createFinalBufferizePass(
      /*alignment=*/64,
      mlir::kernel_gen::transforms::populateExtraBufferizeDialects,
      mlir::kernel_gen::transforms::populateExtraBufferizePatterns));

  if (failed(pm.run(module))) {
    return tensorflow::errors::Internal(
        "Lowering TF to JIT invocation failed.");
  }
  return OkStatus();
}

Status LowerTFtoLoops(mlir::ModuleOp module, llvm::ArrayRef<int64_t> tile_sizes,
                      llvm::ArrayRef<int64_t> unroll_factors,
                      int64_t max_supported_rank, bool enable_ftz,
                      bool index_64bit, bool jit_i64_indexed_for_large_tensors,
                      bool apply_cl_options, const AutoFusionOption &af_option,
                      mlir::DefaultTimingManager &tm) {
  auto timing = tm.getRootScope();
  mlir::PassManager pm(module.getContext());
  pm.enableTiming(timing);

  std::string errorMessage;
  auto output = mlir::openOutputFile("predict_online_funcs_LowerTFtoLoops.mlir", &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    std::terminate();
  }
  output->keep();

  mlir::OpPrintingFlags flag{};
  pm.enableIRPrinting(
    /*shouldPrintBeforePass=*/[](mlir::Pass* p, mlir::Operation* op) {
      return true;
    },
    /*shouldPrintAfterPass=*/[](mlir::Pass* p, mlir::Operation * op) {
      return true;
    },
    /*printModuleScope=*/false, 
    /*printAfterOnlyOnChange=*/false,
    /*printAfterOnlyOnFailure=*/false, 
    output->os(), flag
  );

  if (apply_cl_options)
    applyTensorflowAndCLOptions(pm);
  if (jit_i64_indexed_for_large_tensors) {
    pm.addNestedPass<FuncOp>(
        mlir::kernel_gen::transforms::CreateTFToJITInvocationPass(
            tile_sizes, unroll_factors, max_supported_rank, enable_ftz,
            index_64bit,
            /*jit_i64_indexed_for_large_tensors=*/true));
  }
  pm.addPass(
    mlir::tfext_kernel_gen::createAFHandleFallbackPass(af_option.cache_dir));
  pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeTFNoFallbackPass(true));
  pm.addNestedPass<FuncOp>(mlir::mhlo::createRankSpecializationClusterPass());
  pm.addNestedPass<FuncOp>(
      mlir::mhlo::createRankSpecializationToSCFPass(max_supported_rank));
  pm.addNestedPass<FuncOp>(mlir::mhlo::createChloLegalizeToHloPass());

  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createShapeSimplification());
  // FIXME(zfc)
  // merge assuming op may cost too much
  // pm.addNestedPass<FuncOp>(mlir::createRemoveShapeConstraintsPass());
  pm.addNestedPass<FuncOp>(
      mlir::tfext_kernel_gen::createMergeAssumingLimitOpsPass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(mlir::mhlo::createBroadcastPropagationPass());
  pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());

  // custom pass here
  if (!af_option.static_shape) {
    pm.addNestedPass<FuncOp>(
        mlir::tfext_kernel_gen::createAddFakeSymbolicShapePass());
    pm.addNestedPass<FuncOp>(
        mlir::tfext_kernel_gen::createCanonicalizeExtWithConstraintsPass());
    pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(mlir::createCSEPass());
    pm.addNestedPass<FuncOp>(
        mlir::tfext_kernel_gen::createDelFakeSymbolicShapePass());
    pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  }
  //  TODO(zfc)
  //  useless
  //  pm.addNestedPass<FuncOp>(mlir::tfext_kernel_gen::createFlatAddnComputePass());

  // Transform HLO operations to LinAlg and standard.

  pm.addNestedPass<FuncOp>(::mlir::mhlo::createLegalizeHloToLinalgPass());
  pm.addNestedPass<FuncOp>(
      ::mlir::tfext_kernel_gen::createCustomHloLegalizeToLinalgPass());

  pm.addPass(::mlir::mhlo::createLegalizeToArithmeticPass());
  pm.addNestedPass<FuncOp>(
      mlir::mhlo::createLegalizeHloShapeOpsToStandardPass());

  // Remove the remaining references to unsigned types after all HLO compute
  // operations were converted.
  pm.addPass(mlir::mhlo::createConvertToSignlessPass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());

  // Convert operations from the Complex dialect to the Standard/Math dialects.
  pm.addNestedPass<FuncOp>(::mlir::createConvertComplexToStandardPass());

  // Fuse linalg operations.
  pm.addPass(mlir::memref::createResolveRankedShapeTypeResultDimsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(
      mlir::tfext_kernel_gen::createCustomLinalgElementwiseOpFusionPass());

  //// Partial bufferization: Transforms inparticular HLO and Linalg operations
  /// to / their corresponding LHLO operations and converts the function
  /// signature. / Leaves shape operations untouched.
  ////
  //// TODO(pifon): Rename the pass to CreateHloLinalgBufferizePass or bufferize
  //// in 2 steps: first Linalg, then Hlo. That would need refactoring of
  //// BufferizeTypeConverter.
  pm.addNestedPass<FuncOp>(
      mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(
      mlir::tfext_kernel_gen::createCustomComputeOpAndFuncBufferizePass());
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  // Remove copies which are introduced by canonicalizing
  // BufferCastOp(TensorLoadOp).
  pm.addNestedPass<FuncOp>(mlir::tfext_kernel_gen::createCopyCleanupPass());
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(mlir::tfext_kernel_gen::createCopyImplPass());
  if (af_option.float_compute) {
    pm.addNestedPass<FuncOp>(mlir::tfext_kernel_gen::createFloatComputePass());
  }
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  // Find candidates for buffer reuse. This is only successful if buffer size
  // equality can be determined based on `linalg.generic` operations.
  //

  // TODO(zfc)
  // core dump, and this pass is useless for us
  // pm.addNestedPass<FuncOp>(
  //    mlir::kernel_gen::transforms::CreateBufferReusePass());

  // Approximate Tanh using standard operations.
  pm.addNestedPass<FuncOp>(
      ::mlir::mhlo::createLegalizeTrigonometricToApproximationPass());
  // Transform the Linalg ops inside of the loop nest into parallel loops.
  pm.addNestedPass<FuncOp>(::mlir::createConvertLinalgToParallelLoopsPass());
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(
      ::mlir::tfext_kernel_gen::createAFReduceFusionInitAndEpiPass());
  pm.addNestedPass<FuncOp>(
      ::mlir::tfext_kernel_gen::createAFReduceConvertPass());
  pm.addNestedPass<FuncOp>(
      ::mlir::tfext_kernel_gen::createAFReduceSpecOptPass());
  pm.addNestedPass<FuncOp>(
      ::mlir::tfext_kernel_gen::createAFReduceWarpTilePass());

  // Canonicalize the code to simplify index computations. This is needed so
  // that loop bounds have the same value.
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  // Run CSE to ensure that loads and stores to the same subview get
  // recognized as such.
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  // Collapse and tile parallel loops for GPU only.
  pm.addNestedPass<FuncOp>(mlir::createCollapseParallelLoopsTo1DPass());

  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(mlir::bufferization::createPromoteBuffersToStackPass(
      [](Value alloc) { return IsSmallAlloc(alloc); }));
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(
      ::mlir::tfext_kernel_gen::createAFMarkHostParallelOpPass());
  pm.addNestedPass<FuncOp>(
      mlir::createTileLoopsPass(tile_sizes, unroll_factors));
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(mlir::tfext_kernel_gen::createMergeSCFPass());
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());

  // fuse multiple kernel launch into one op
  pm.addPass(mlir::tfext_kernel_gen::createFuseKernelLaunchPass(
      af_option.fuse_info_file));
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  pm.enableIRPrinting(
      [](mlir::Pass *pass, mlir::Operation *) {
        // if (pass->getName() == "CopyImplPass") {
        //   return true;
        // }
        return false;
      },
      [](mlir::Pass *pass, mlir::Operation *) {
        // if (pass->getName() == "CopyImplPass") {
        //   return true;
        // }
        return false;
      },
      false, false, false);
  if (failed(pm.run(module))) {
    return tensorflow::errors::Internal("Lowering TF to loops failed.");
  }
  return OkStatus();
}

Status LowerLoopsToGPU(mlir::ModuleOp module, bool embed_memref_prints,
                       bool index_64bit, bool apply_cl_options,
                       const AutoFusionOption &af_option,
                       mlir::DefaultTimingManager &tm) {
  mlir::PassManager pm(module.getContext());
  auto timing = tm.getRootScope();
  pm.enableTiming(timing);

  std::string errorMessage;
  auto output = mlir::openOutputFile("predict_online_funcs_LowerLoopsToGPU.mlir", &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    std::terminate();
  }
  output->keep();

  mlir::OpPrintingFlags flag{};
  pm.enableIRPrinting(
    /*shouldPrintBeforePass=*/[](mlir::Pass* p, mlir::Operation* op) {
      return true;
    },
    /*shouldPrintAfterPass=*/[](mlir::Pass* p, mlir::Operation * op) {
      return true;
    },
    /*printModuleScope=*/false, 
    /*printAfterOnlyOnChange=*/false,
    /*printAfterOnlyOnFailure=*/false, 
    output->os(), flag
  );

  if (apply_cl_options)
    applyTensorflowAndCLOptions(pm);

  // Greedily map the remaining loop to GPU hardware dimensions.
  if (!af_option.only_cpu) {
    pm.addNestedPass<FuncOp>(mlir::createGpuMapParallelLoopsPass());
  }

  // Expand memref_reshape to its ranked form so that we can propagate
  // scalars and avoid allocation.
  //
  // TODO(zfc)
  // it's safe to remove ArithExpandOps pass?
  // pm.addNestedPass<FuncOp>(mlir::arith::createArithExpandOpsPass());
  pm.addNestedPass<FuncOp>(mlir::memref::createExpandOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::kernel_gen::transforms::CreateShapeToDescriptorsPass());
  // Before bufferizing further, remove unused tensor_to_memref, so that we do
  // not create allocations for tensor computations that are not actually
  // needed.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());
  // Before inserting more allocs, map the ones we already have to the
  // tf runtime. That ensures that all allocations for the actual computation
  // end up on the device, whereas allocations for shape computation and host
  // side things remain on the host.
  // Longer term, this should be handled by proper device placement.
  pm.addPass(mlir::kernel_gen::tf_framework::CreateEmbedTFFrameworkPass());
  pm.addNestedPass<FuncOp>(
      mlir::tfext_kernel_gen::createAFCheckExternalCallResultPass());
  // Now lower the shape computations, bufferize all remaining ops and insert
  // deallocs.
  pm.addPass(mlir::createFinalBufferizePass(
      /*alignment=*/64,
      mlir::kernel_gen::transforms::populateExtraBufferizeDialects,
      mlir::kernel_gen::transforms::populateExtraBufferizePatterns));
  pm.addNestedPass<FuncOp>(::mlir::bufferization::createBufferHoistingPass());
  // Free all temporaries,
  pm.addNestedPass<FuncOp>(
      ::mlir::bufferization::createBufferDeallocationPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // pm.addPass(
  //   mlir::tfext_kernel_gen::createBatchComputeFusionPass());
  // pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  // pm.addNestedPass<FuncOp>(::mlir::createCSEPass());

  // Apply the mapping and go to GPU. We cannot do this earlier as the GPU
  // dialect requires memrefs.
  if (!af_option.only_cpu) {
    pm.addNestedPass<FuncOp>(
        mlir::tfext_kernel_gen::createCustomParallelLoopsToGpuPass());
  }

  // Some basic cleanup.
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(::mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(
      mlir::tfext_kernel_gen::createAFGPUFrontendOptPass());
  // Make loops with min bounds into a conditional plus static bounds.
  pm.addNestedPass<FuncOp>(mlir::createForLoopSpecializationPass());
  // Take launches to launches with kernels.
  pm.addPass(mlir::createGpuLauchSinkIndexComputationsPass());
  const std::string gpuDataLayoutSpec =
      index_64bit ? "#dlti.dl_spec<#dlti.dl_entry<index,64:i64>>"
                  : "#dlti.dl_spec<#dlti.dl_entry<index,32:i32>>";
  pm.addPass(mlir::createGpuKernelOutliningPass(gpuDataLayoutSpec));
  pm.addPass(mlir::tfext_kernel_gen::createAFGPUAllocaToGPUMemPass());

  pm.addPass(::mlir::createLowerAffinePass());
  // Constraints are removed as late as possible and before lowering to CFG.
  pm.addNestedPass<FuncOp>(::mlir::createConvertShapeConstraintsPass());
  pm.addNestedPass<FuncOp>(::mlir::createCanonicalizerPass());
  pm.addPass(::mlir::createConvertSCFToCFPass());
  // Map asserts to the tensorflow framework.
  pm.addPass(mlir::kernel_gen::tf_framework::CreateRewriteTFFrameworkAssert());
  if (embed_memref_prints) {
    pm.addPass(mlir::kernel_gen::transforms::CreateEmbedMemRefPrintsPass());
  }
  pm.addPass(mlir::tfext_kernel_gen::createInterleaveLoadAndComputePass());
  if (failed(pm.run(module))) {
    return tensorflow::errors::Internal("Lowering to GPU kernels failed.");
  }
  return OkStatus();
}

Status LowerKernelBodiesToLowLevelIr(mlir::ModuleOp module,
                                     bool apply_cl_options,
                                     mlir::DefaultTimingManager &tm) {
  mlir::PassManager pm(module.getContext());
  auto timing = tm.getRootScope();
  pm.enableTiming(timing);
  // We cannot verify as the signature of the kernel is rewritten.
  // pm.enableVerifier(false);
  if (apply_cl_options)
    tensorflow::applyTensorflowAndCLOptions(pm);
  auto &kernelPm = pm.nest<::mlir::gpu::GPUModuleOp>();
  kernelPm.addPass(::mlir::createConvertSCFToCFPass());

  kernelPm.addPass(mlir::createGpuKernelToNvvmPass());
  kernelPm.addPass(mlir::NVVM::createOptimizeForTargetPass());
  // Remove all location information to prevent a debug build.
  pm.addPass(::mlir::createStripDebugInfoPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (failed(pm.run(module))) {
    return tensorflow::errors::Internal(
        "Lowering to low-level device IR failed.");
  }

  return OkStatus();
}

Status AmendKernelLLVMIRWithStaticKnowledge(mlir::ModuleOp module,
                                            bool apply_cl_options,
                                            mlir::DefaultTimingManager &tm) {
  mlir::PassManager pm(module.getContext());
  auto timing = tm.getRootScope();
  pm.enableTiming(timing);
  if (apply_cl_options)
    applyTensorflowAndCLOptions(pm);

  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreatePropagateShapeKnowledgeToKernels());
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreatePropagateTfAbiKnowledgeToKernels());
  return failed(pm.run(module))
             ? tensorflow::errors::Internal(
                   "Amending LLVMIR with static knowledge failed.")
             : OkStatus();
}

Status BeforeGneratedDeviceCode(mlir::ModuleOp module, bool apply_cl_options,
                                int ptr_size_threshold,
                                mlir::DefaultTimingManager &tm) {
  mlir::PassManager pm(module.getContext());
  auto timing = tm.getRootScope();
  pm.enableTiming(timing);
  if (apply_cl_options)
    applyTensorflowAndCLOptions(pm);
  pm.addPass(mlir::tfext_kernel_gen::createReduceKernelParamsPass());
  pm.addPass(mlir::tfext_kernel_gen::createPackParamsPass(ptr_size_threshold));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::tfext_kernel_gen::createAFIndexCanonicalizePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  return failed(pm.run(module))
             ? tensorflow::errors::Internal(
                   "Amending LLVMIR with static knowledge failed.")
             : OkStatus();
}

Status GenerateDeviceCode(mlir::ModuleOp module,
                          llvm::StringRef gpu_binary_attr_name,
                          llvm::ArrayRef<std::string> architectures,
                          bool print_ptx, bool print_llvmir, bool enable_ftz,
                          bool apply_cl_options,
                          mlir::DefaultTimingManager &tm) {
  mlir::PassManager pm(module.getContext());
  auto timing = tm.getRootScope();
  pm.enableTiming(timing);
  if (apply_cl_options)
    applyTensorflowAndCLOptions(pm);
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto &kernel_pm = pm.nest<mlir::gpu::GPUModuleOp>();
  // Remove debug information to ensure we do not create debug PTX.
  kernel_pm.addPass(mlir::createStripDebugInfoPass());
  kernel_pm.addPass(mlir::kernel_gen::transforms::CreateGpuKernelToBlobPass(
      gpu_binary_attr_name, architectures, print_ptx, print_llvmir,
      enable_ftz));

  return failed(pm.run(module))
             ? tensorflow::errors::Internal("Generating device code failed.")
             : OkStatus();
}

Status LowerHostSideToFinalForm(mlir::ModuleOp module, bool apply_cl_options,
                                mlir::DefaultTimingManager &tm) {
  mlir::PassManager pm(module.getContext());
  auto timing = tm.getRootScope();
  pm.enableTiming(timing);
  if (apply_cl_options)
    applyTensorflowAndCLOptions(pm);

  pm.addPass(mlir::kernel_gen::transforms::CreateTFKernelToLLVMPass(
      kGpuBinaryAttrName));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::tfext_kernel_gen::createCIfacePass());

  return failed(pm.run(module)) ? tensorflow::errors::Internal(
                                      "Final lowering of host side failed.")
                                : OkStatus();
}

} // namespace

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SetupContextAndParseModule(mlir::MLIRContext &context,
                           llvm::StringRef tf_code) {
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  registry.insert<
      mlir::chlo::ChloDialect, mlir::mhlo::MhloDialect,
      mlir::memref::MemRefDialect, mlir::scf::SCFDialect, mlir::AffineDialect,
      mlir::tensor::TensorDialect, mlir::bufferization::BufferizationDialect,
      mlir::linalg::LinalgDialect, mlir::auto_fusion::AutoFusionDialect>();
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  context.appendDialectRegistry(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(tf_code, &context);
  if (!module)
    return tensorflow::Status(tensorflow::error::Code::INVALID_ARGUMENT,
                              "invalid kernel IR");
  return module;
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GenerateKernelForTfCode(
    mlir::MLIRContext &context, llvm::StringRef tf_code,
    llvm::ArrayRef<std::string> architectures,
    llvm::ArrayRef<int64_t> tile_sizes, llvm::ArrayRef<int64_t> unroll_factors,
    int64_t max_supported_rank, bool embed_memref_prints, bool print_ptx,
    bool print_llvmir, bool enable_ftz, bool index_64bit, bool,
    bool jit_i64_indexed_for_large_tensors, bool apply_cl_options,
    AutoFusionOption af_option) {
  mlir::auto_fusion::metrics::Metrics::ins().setCurName("compile");
  OPS_AF_METRICS("env", "AF_FLOAT_COMPUTE", af_option.float_compute);
  OPS_AF_METRICS("env", "AF_V2", mlir::auto_fusion::enableAFV2());
  OPS_AF_METRICS("env", "AF_DENSE_FUSION",
                 mlir::auto_fusion::enableAFDenseFusion());
  OPS_AF_METRICS("env", "AF_LIMIT_FUSION",
                 mlir::auto_fusion::enableAFLimitFusion());
  OPS_AF_METRICS("env", "AF_ENABLE_ALGO_CACHE",
                 mlir::auto_fusion::enableAFAlgoCache());
  OPS_AF_METRICS("env", "AF_CLUSTER_FUSION",
                 bool(af_option.fuse_info_file.size()));
  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      SetupContextAndParseModule(context, tf_code));
  mlir::DefaultTimingManager tm;
  tm.setEnabled(true);

  TF_RETURN_IF_ERROR(LowerTFtoLoops(module.get(), tile_sizes, unroll_factors,
                                    max_supported_rank, enable_ftz, index_64bit,
                                    jit_i64_indexed_for_large_tensors,
                                    apply_cl_options, af_option, tm));
  TF_RETURN_IF_ERROR(LowerLoopsToGPU(module.get(), embed_memref_prints,
                                     index_64bit, apply_cl_options, af_option,
                                     tm));
  TF_RETURN_IF_ERROR(
      LowerKernelBodiesToLowLevelIr(module.get(), apply_cl_options, tm));
  TF_RETURN_IF_ERROR(
      AmendKernelLLVMIRWithStaticKnowledge(module.get(), apply_cl_options, tm));
  TF_RETURN_IF_ERROR(BeforeGneratedDeviceCode(
      module.get(), apply_cl_options, af_option.ptr_size_threshold, tm));
  TF_RETURN_IF_ERROR(GenerateDeviceCode(module.get(), kGpuBinaryAttrName,
                                        architectures, print_ptx, print_llvmir,
                                        enable_ftz, apply_cl_options, tm));
  TF_RETURN_IF_ERROR(
      LowerHostSideToFinalForm(module.get(), apply_cl_options, tm));
  std::string timingLog;
  llvm::raw_string_ostream oss(timingLog);
  tm.dumpAsTree(oss);
  oss.flush();
  auto split = [](const std::string &input, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream ss(input);
    std::string part;
    while (std::getline(ss, part, delimiter)) {
      parts.push_back(part);
    }

    return parts;
  };

  OPS_AF_METRICS("timing", "report", split(timingLog, '\n'));
  if (af_option.metric_file.size()) {
    mlir::auto_fusion::metrics::Metrics::ins().dumpToFile(
        af_option.metric_file);
  } else {
    llvm::outs() << mlir::auto_fusion::metrics::Metrics::ins().dump() << "\n";
  }
  llvm::outs() << mlir::auto_fusion::metrics::Metrics::ins().dump() << "\n";
  tm.setEnabled(false);
  return module;
}

} // namespace kernel_gen
} // namespace tensorflow
