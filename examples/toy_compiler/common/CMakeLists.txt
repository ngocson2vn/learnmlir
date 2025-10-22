# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls)
add_public_tablegen_target(ToyCommonPassIncGen)

add_library(common
  passes.cc
)
add_dependencies(common 
  ToyCommonPassIncGen
  MLIRIR
  MLIRDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRLinalgTransforms
  MLIRGPUTransforms
  MLIRSCFToControlFlow
  MLIRParser
  MLIRSupport
  MLIRTableGen
  clang_rt.ubsan_standalone-x86_64
)
target_compile_options(common PRIVATE -fno-rtti -g -O0)
