get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# triton-opt
add_llvm_executable(triton-opt triton-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-opt)
target_link_libraries(triton-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  # tests
  # TritonTestAnalysis
  # TritonTestDialect
  # TritonAMDGPUTestAnalysis
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-opt)


# triton_compiler
add_llvm_executable(triton_compiler triton_compiler.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(triton_compiler)
target_link_libraries(triton_compiler PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}

  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms

  # LLVM
  LLVMTarget
  LLVMX86CodeGen
  LLVMX86AsmParser
  LLVMX86Disassembler
  LLVMX86TargetMCA
  LLVMX86Desc
  LLVMX86Info
)

mlir_check_all_link_libraries(triton_compiler)
