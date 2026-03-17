set(LLVM_TARGET_DEFINITIONS TritonDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=tt)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=tt)

set(LLVM_TARGET_DEFINITIONS TritonTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(TritonDialectIncGen)

add_library(TritonIR
  Dialect.cpp
  Types.cpp
)
add_dependencies(TritonIR
  TritonDialectIncGen
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