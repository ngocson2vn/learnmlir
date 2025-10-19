# Generate operation definitions
set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls)
add_public_tablegen_target(ToyBackendPassIncGen)

include_directories(${CUDA_DIR}/include)

add_library(backend
  backend.cc
  utils.cc
  llvm_utils.cc
  cuda_utils.cc
  tile_loops_pass.cc
  nvidia_backend.cc
  passes.cc
)
add_dependencies(backend 
  ToyBackendPassIncGen
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
target_compile_options(backend PRIVATE -fno-rtti -g -O0)
